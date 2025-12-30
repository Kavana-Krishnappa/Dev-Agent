import requests
import json
import time
from typing import Dict, List, Optional

class LLMCodeAnalyzer:
    """Free local LLM integration using Ollama + CodeLlama"""
    
    def __init__(self, base_url: str = "http://localhost:11434"):
        self.base_url = base_url
        self.model = "codellama:7b"
        self.ollama_available = self._check_ollama()
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is running"""
        try:
            response = requests.get(f"{self.base_url}/api/tags", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def analyze_code_semantics(self, code: str, file_path: str, language: str) -> dict:
        """Deep semantic analysis using LLM"""
        if not self.ollama_available:
            return {"semantic_issues": [], "llm_available": False, "raw_analysis": ""}
        
        prompt = f"""Analyze this {language} code for REAL issues (not just style):

FILE: {file_path}
CODE:
{code}

Find:
1. Logic errors (infinite loops, null pointers, off-by-one)
2. Security vulnerabilities (exploitable, not just patterns)
3. Performance bottlenecks
4. Design issues (god classes, tight coupling)
5. Missing error handling

Format each issue as:
LINE: <number>
SEVERITY: Critical|High|Medium|Low
TYPE: Logic|Security|Performance|Design|ErrorHandling
ISSUE: <description>
IMPACT: <what breaks>
FIX: <how to fix>
---

Be specific. Only report REAL issues."""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 1000}
                },
                timeout=60
            )
            
            if response.status_code == 200:
                result = response.json()
                raw = result.get("response", "")
                issues = self._parse_llm_issues(raw, file_path)
                return {
                    "semantic_issues": issues,
                    "llm_available": True,
                    "raw_analysis": raw
                }
        except Exception as e:
            print(f"LLM analysis failed: {e}")
        
        return {"semantic_issues": [], "llm_available": False, "raw_analysis": ""}
    
    def verify_requirement_implementation(self, code: str, requirement: dict) -> dict:
        """Smart requirement verification using LLM"""
        if not self.ollama_available:
            return {"implemented": "Unknown", "confidence": 0, "evidence": [], "missing": []}
        
        req_text = requirement.get("text", "")
        req_id = requirement.get("id", "")
        
        prompt = f"""Does this code ACTUALLY implement this requirement?

REQUIREMENT {req_id}: {req_text}

CODE:
{code}

Answer:
IMPLEMENTED: Yes|Partial|No
CONFIDENCE: 0-100
EVIDENCE: <specific code that implements it>
MISSING: <what's not implemented>
ISSUES: <problems with implementation>

Remember: Comments don't count. Only working code counts."""

        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": 0.1, "num_predict": 500}
                },
                timeout=30
            )
            
            if response.status_code == 200:
                result = response.json()
                raw = result.get("response", "")
                return self._parse_llm_verification(raw)
        except Exception as e:
            print(f"LLM verification failed: {e}")
        
        return {"implemented": "Unknown", "confidence": 0, "evidence": [], "missing": []}
    
    def _parse_llm_issues(self, raw_text: str, file_path: str) -> List[Dict]:
        """Parse LLM output into structured issues"""
        issues = []
        current_issue = {}
        
        for line in raw_text.split('\n'):
            line = line.strip()
            
            if line.startswith('LINE:'):
                if current_issue:
                    issues.append(current_issue)
                current_issue = {
                    "category": "AI-Detected",
                    "file": file_path,
                    "line": self._extract_number(line)
                }
            elif line.startswith('SEVERITY:'):
                current_issue["severity"] = line.split(':', 1)[1].strip()
            elif line.startswith('TYPE:'):
                current_issue["type"] = line.split(':', 1)[1].strip()
            elif line.startswith('ISSUE:'):
                current_issue["explanation"] = line.split(':', 1)[1].strip()
            elif line.startswith('IMPACT:'):
                current_issue["impact"] = line.split(':', 1)[1].strip()
            elif line.startswith('FIX:'):
                current_issue["suggested_fix"] = line.split(':', 1)[1].strip()
            elif line == '---' and current_issue:
                issues.append(current_issue)
                current_issue = {}
        
        if current_issue:
            issues.append(current_issue)
        
        return issues
    
    def _parse_llm_verification(self, raw_text: str) -> Dict:
        """Parse LLM verification response"""
        result = {
            "implemented": "Unknown",
            "confidence": 0,
            "evidence": [],
            "missing": [],
            "issues": []
        }
        
        for line in raw_text.split('\n'):
            line = line.strip()
            
            if line.startswith('IMPLEMENTED:'):
                value = line.split(':', 1)[1].strip()
                if 'yes' in value.lower():
                    result["implemented"] = "Yes"
                elif 'partial' in value.lower():
                    result["implemented"] = "Partial"
                elif 'no' in value.lower():
                    result["implemented"] = "No"
            
            elif line.startswith('CONFIDENCE:'):
                try:
                    result["confidence"] = int(self._extract_number(line))
                except:
                    pass
            
            elif line.startswith('EVIDENCE:'):
                result["evidence"].append(line.split(':', 1)[1].strip())
            
            elif line.startswith('MISSING:'):
                result["missing"].append(line.split(':', 1)[1].strip())
            
            elif line.startswith('ISSUES:'):
                result["issues"].append(line.split(':', 1)[1].strip())
        
        return result
    
    def _extract_number(self, text: str) -> int:
        """Extract first number from text"""
        import re
        match = re.search(r'\d+', text)
        return int(match.group()) if match else 0