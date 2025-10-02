# parsers/multi_modal_parser.py
import fitz  # PyMuPDF
import cv2
import numpy as np
from PIL import Image
from typing import List, Dict, Any, Optional
import pytesseract
from transformers import AutoProcessor, AutoModel
from main import DocumentType

class MultiModalParser:
    def __init__(self):
        # Load vision models for zero-shot layout understanding
        self.layout_model = AutoModel.from_pretrained("microsoft/layoutlmv3-base")
        self.processor = AutoProcessor.from_pretrained("microsoft/layoutlmv3-base")

    async def _parse_digital_pdf(self, file_path: str) -> List[Dict]:
        """Parse digital PDF with native text extraction"""
        doc = fitz.open(file_path)
        results = []
        
        for page_num, page in enumerate(doc):
            # Extract text with bounding boxes
            blocks = page.get_text("dict")
            
            for block in blocks["blocks"]:
                if "lines" in block:  # Text block
                    block_text = ""
                    block_bbox = None
                    
                    for line in block["lines"]:
                        line_text = ""
                        line_bbox = None
                        
                        for span in line["spans"]:
                            line_text += span["text"]
                            span_bbox = (span["bbox"][0], span["bbox"][1], 
                                    span["bbox"][2], span["bbox"][3])
                            if line_bbox is None:
                                line_bbox = span_bbox
                            else:
                                # Expand bounding box
                                line_bbox = (min(line_bbox[0], span_bbox[0]),
                                        min(line_bbox[1], span_bbox[1]),
                                        max(line_bbox[2], span_bbox[2]),
                                        max(line_bbox[3], span_bbox[3]))
                        
                        block_text += line_text + " "
                        if block_bbox is None:
                            block_bbox = line_bbox
                        else:
                            # Expand block bounding box
                            block_bbox = (min(block_bbox[0], line_bbox[0]),
                                        min(block_bbox[1], line_bbox[1]),
                                        max(block_bbox[2], line_bbox[2]),
                                        max(block_bbox[3], line_bbox[3]))
                    
                    if block_text.strip():
                        results.append({
                            'text': block_text.strip(),
                            'type': 'text',
                            'bbox': block_bbox,
                            'page_num': page_num,
                            'confidence': 0.95
                        })
                
                elif "image" in block:  # Image block
                    results.append({
                        'text': '',
                        'type': 'image',
                        'bbox': block['bbox'],
                        'page_num': page_num,
                        'confidence': 0.8
                    })
        
        return results

    async def _parse_image(self, file_path: str) -> List[Dict]:
        """Parse image file with OCR"""
        img = Image.open(file_path)
        
        # Extract text with bounding boxes
        ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
        
        # Extract layout elements
        layout_elements = await self._extract_layout_elements(img, 0)
        
        # Combine OCR and layout information
        results = self._merge_ocr_layout(ocr_data, layout_elements, 0)
        
        return results

    async def _parse_generic(self, file_path: str) -> List[Dict]:
        """Fallback parsing method"""
        try:
            # Try as PDF first
            return await self._parse_digital_pdf(file_path)
        except:
            try:
                # Try as image
                return await self._parse_image(file_path)
            except:
                # Return empty if all fails
                return []

    def _merge_ocr_layout(self, ocr_data: Dict, layout_elements: List[Dict], page_num: int) -> list[dict]:
        """Merge OCR data with layout elements"""
        results = []
        
        # Process OCR data
        n_boxes = len(ocr_data['text'])
        for i in range(n_boxes):
            if int(ocr_data['conf'][i]) > 30:  # Confidence threshold
                text = ocr_data['text'][i]
                if text.strip():
                    bbox = (ocr_data['left'][i], ocr_data['top'][i],
                        ocr_data['left'][i] + ocr_data['width'][i],
                        ocr_data['top'][i] + ocr_data['height'][i])
                    
                    results.append({
                        'text': text,
                        'type': 'text',
                        'bbox': bbox,
                        'page_num': page_num,
                        'confidence': int(ocr_data['conf'][i]) / 100.0
                    })
        
        # Add layout elements that don't have text
        for element in layout_elements:
            if element['type'] != 'text_block':  # Already covered by OCR
                results.append({
                    'text': '',
                    'type': element['type'],
                    'bbox': element['bbox'],
                    'page_num': page_num,
                    'confidence': element['confidence']
                })
        
        return results

    async def parse_document(self, file_path: str) -> List[Dict]:
        """Parse document using multiple approaches based on content type"""
        
        # Detect document type
        doc_type = self._detect_document_type(file_path)
        
        if doc_type == DocumentType.SCANNED_PDF:
            return await self._parse_scanned_pdf(file_path)
        elif doc_type == DocumentType.DIGITAL_PDF:
            return await self._parse_digital_pdf(file_path)
        elif doc_type == DocumentType.IMAGE:
            return await self._parse_image(file_path)
        else:
            return await self._parse_generic(file_path)
    
    def _detect_document_type(self, file_path: str) -> DocumentType:
        """Intelligent document type detection"""
        try:
            doc = fitz.open(file_path)
            # Check if document has selectable text
            text_content = ""
            for page in doc:
                text_content += page.get_text()
            
            if len(text_content.strip()) > 100:
                return DocumentType.DIGITAL_PDF
            else:
                return DocumentType.SCANNED_PDF
        except:
            return DocumentType.IMAGE
    
    async def _parse_scanned_pdf(self, file_path: str) -> List[Dict]:
        """Parse scanned PDF with OCR and layout analysis"""
        doc = fitz.open(file_path)
        results = []
        
        for page_num, page in enumerate(doc):
            # Convert to image
            mat = fitz.Matrix(2.0, 2.0)  # 2x zoom for better OCR
            pix = page.get_pixmap(matrix=mat)
            img_data = pix.tobytes("png")
            img = Image.open(io.BytesIO(img_data))
            
            # Extract text with bounding boxes
            ocr_data = pytesseract.image_to_data(img, output_type=pytesseract.Output.DICT)
            
            # Extract layout elements
            layout_elements = await self._extract_layout_elements(img, page_num)
            
            # Combine OCR and layout information
            page_data = self._merge_ocr_layout(ocr_data, layout_elements, page_num)
            results.extend(page_data)
            
        return results
    
    async def _extract_layout_elements(self, image: Image.Image, page_num: int) -> List[Dict]:
        """Extract layout elements using computer vision"""
        # Convert PIL to OpenCV
        opencv_image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Detect various layout elements
        elements = []
        
        # 1. Text blocks
        text_blocks = self._detect_text_blocks(opencv_image)
        elements.extend(text_blocks)
        
        # 2. Tables
        tables = await self._detect_tables(opencv_image)
        elements.extend(tables)
        
        # 3. Form fields
        form_fields = await self._detect_form_fields(opencv_image)
        elements.extend(form_fields)
        
        # 4. Images
        images = self._detect_images(opencv_image)
        elements.extend(images)
        
        return elements
    
    def _detect_text_blocks(self, image: np.ndarray) -> List[Dict]:
        """Detect text blocks using MSER and contour analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Use MSER for text region detection
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        
        text_blocks = []
        for region in regions:
            x, y, w, h = cv2.boundingRect(region)
            # Filter by size and aspect ratio
            if w > 20 and h > 10 and w/h < 10:
                text_blocks.append({
                    'type': 'text_block',
                    'bbox': (x, y, x+w, y+h),
                    'confidence': 0.8
                })
        
        return text_blocks
    
    async def _detect_tables(self, image: np.ndarray) -> List[Dict]:
        """Detect tables using line detection and structure analysis"""
        # Convert to grayscale
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Apply threshold
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY_INV)
        
        # Detect horizontal and vertical lines
        horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (40, 1))
        vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 40))
        
        horizontal_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel, iterations=2)
        vertical_lines = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel, iterations=2)
        
        # Combine lines
        table_structure = cv2.addWeighted(horizontal_lines, 0.5, vertical_lines, 0.5, 0.0)
        
        # Find contours
        contours, _ = cv2.findContours(table_structure, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        tables = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            # Filter by size (tables are usually large)
            if w > 100 and h > 50:
                tables.append({
                    'type': 'table',
                    'bbox': (x, y, x+w, y+h),
                    'confidence': 0.9
                })
        
        return tables
    
    async def _detect_form_fields(self, image: np.ndarray) -> List[Dict]:
        """Detect form fields using template matching and shape analysis"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # Detect rectangles (common form field shapes)
        _, thresh = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY)
        
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        form_fields = []
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            
            # Check if it's likely a form field (specific aspect ratios, sizes)
            aspect_ratio = w / h if h > 0 else 0
            area = w * h
            
            if (20 < w < 300 and 10 < h < 100 and 
                1 < aspect_ratio < 10 and area > 200):
                
                # Check if it's empty (form field characteristic)
                roi = gray[y:y+h, x:x+w]
                white_pixels = np.sum(roi > 200)
                total_pixels = roi.size
                fill_ratio = white_pixels / total_pixels if total_pixels > 0 else 0
                
                if fill_ratio < 0.3:  # Mostly empty - likely form field
                    form_fields.append({
                        'type': 'form_field',
                        'bbox': (x, y, x+w, y+h),
                        'confidence': 0.7
                    })
        
        return form_fields