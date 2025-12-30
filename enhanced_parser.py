"""
Enhanced Requirement Parser - Handles ANY text format
"""

import json
import re
from typing import List, Dict

class EnhancedRequirementParser:
    """Parse requirements from ANY text format"""
    
    def parse(self, text: str) -> Dict:
        text = text.strip()
        
        # Try JSON first
        if text.startswith('{') or text.startswith('['):
            try:
                return self._parse_json(text)
            except:
                pass
        
        # Fall back to text parsing
        return self._parse_freeform_text(text)
    
    def _parse_json(self, json_text: str) -> Dict:
        data = json.loads(json_text)
        
        if isinstance(data, list):
            requirements_list = data
        elif isinstance(data, dict):
            requirements_list = [data]
        else:
            raise ValueError("Invalid JSON")
        
        parsed = []
        for req in requirements_list:
            parsed.append({
                "id": req.get("id", f"REQ-{len(parsed)+1}"),
                "text": req.get("description", req.get("text", "")),
                "type": self._classify_requirement(req.get("description", "")),
            })
        
        return {
            "requirements": parsed,
            "format_detected": "json",
            "total_count": len(parsed)
        }
    
    def _parse_freeform_text(self, text: str) -> Dict:
        requirements = []
        lines = text.split('\n')
        
        # Pattern 1: FR-01, REQ-1, etc.
        explicit_reqs = self._extract_explicit(lines)
        if explicit_reqs:
            requirements.extend(explicit_reqs)
        
        # Pattern 2: 1. 2. 3.
        if not requirements:
            numbered_reqs = self._extract_numbered(lines)
            if numbered_reqs:
                requirements.extend(numbered_reqs)
        
        # Pattern 3: - * •
        if not requirements:
            bullet_reqs = self._extract_bullets(lines)
            if bullet_reqs:
                requirements.extend(bullet_reqs)
        
        # Pattern 4: Paragraphs
        if not requirements:
            para_reqs = self._extract_paragraphs(text)
            if para_reqs:
                requirements.extend(para_reqs)
        
        return {
            "requirements": requirements,
            "format_detected": "text",
            "total_count": len(requirements)
        }
    
    def _extract_explicit(self, lines: List[str]) -> List[Dict]:
        requirements = []
        pattern = r'^(?:FR|REQ|R|NR|NFR|SR|AC|US)[-\s]?(\d+)[:\s.\-]+(.+)$'
        
        for line in lines:
            line = line.strip()
            match = re.match(pattern, line, re.IGNORECASE)
            if match and len(match.group(2)) > 10:
                req_id, req_text = match.groups()
                requirements.append({
                    "id": f"REQ-{req_id}",
                    "text": req_text.strip(),
                    "type": self._classify_requirement(req_text)
                })
        return requirements
    
    def _extract_numbered(self, lines: List[str]) -> List[Dict]:
        requirements = []
        pattern = r'^(\d+)\.[\s]+(.+)$'
        
        for line in lines:
            line = line.strip()
            match = re.match(pattern, line)
            if match and len(match.group(2)) > 10:
                num, req_text = match.groups()
                requirements.append({
                    "id": f"REQ-{num}",
                    "text": req_text.strip(),
                    "type": self._classify_requirement(req_text)
                })
        return requirements
    
    def _extract_bullets(self, lines: List[str]) -> List[Dict]:
        requirements = []
        pattern = r'^[-*•]\s+(.+)$'
        
        for line in lines:
            line = line.strip()
            match = re.match(pattern, line)
            if match and len(match.group(1)) > 10:
                req_text = match.group(1).strip()
                requirements.append({
                    "id": f"REQ-{len(requirements)+1}",
                    "text": req_text,
                    "type": self._classify_requirement(req_text)
                })
        return requirements
    
    def _extract_paragraphs(self, text: str) -> List[Dict]:
        paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
        requirements = []
        
        for i, para in enumerate(paragraphs):
            if len(para) > 20:
                requirements.append({
                    "id": f"REQ-{i+1}",
                    "text": para,
                    "type": self._classify_requirement(para)
                })
        return requirements
    
    def _classify_requirement(self, text: str) -> str:
        text_lower = text.lower()
        
        if any(kw in text_lower for kw in ['auth', 'login', 'password', 'security', 'https', 'encrypt']):
            return 'security'
        elif any(kw in text_lower for kw in ['performance', 'speed', 'fast', 'latency', 'second']):
            return 'performance'
        elif any(kw in text_lower for kw in ['api', 'endpoint', 'rest', 'http']):
            return 'api'
        else:
            return 'functional'