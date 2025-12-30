from typing import Dict, Literal
import hashlib
import json
from pathlib import Path

AnalysisMode = Literal["off", "smart", "always"]

class AnalysisStrategy:
    """Determines when to use LLM vs rule-based analysis"""
    
    def __init__(self, cache_file: str = ".analysis_cache.json"):
        self.cache_file = Path(cache_file)
        self.cache = self._load_cache()
    
    def get_strategy(
        self, 
        code: str, 
        file_path: str, 
        mode: AnalysisMode,
        llm_available: bool
    ) -> Dict[str, bool]:
        """
        Returns: {
            "use_rules": bool,
            "use_llm": bool,
            "use_llm_verification": bool,
            "reason": str
        }
        """
        file_size = len(code.split('\n'))
        
        # LLM unavailable - fallback to rules
        if not llm_available:
            return {
                "use_rules": True,
                "use_llm": False,
                "use_llm_verification": False,
                "reason": "LLM unavailable - using rule-based analysis"
            }
        
        # User disabled AI
        if mode == "off":
            return {
                "use_rules": True,
                "use_llm": False,
                "use_llm_verification": False,
                "reason": "AI analysis disabled by user"
            }
        
        # User wants AI always
        if mode == "always":
            return {
                "use_rules": True,
                "use_llm": True,
                "use_llm_verification": True,
                "reason": "AI analysis forced by user"
            }
        
        # Smart mode - decide based on file characteristics
        is_critical = self._is_critical_file(file_path)
        is_complex = self._is_complex_code(code)
        
        # Small utility files - rules only
        if file_size < 50 and not is_critical:
            return {
                "use_rules": True,
                "use_llm": False,
                "use_llm_verification": False,
                "reason": f"Small file ({file_size} lines) - rule-based sufficient"
            }
        
        # Critical files - always use LLM
        if is_critical:
            return {
                "use_rules": True,
                "use_llm": True,
                "use_llm_verification": True,
                "reason": "Critical security/business logic file"
            }
        
        # Medium complexity - rules + LLM verification
        if 200 <= file_size <= 1000:
            return {
                "use_rules": True,
                "use_llm": False,
                "use_llm_verification": True,
                "reason": f"Medium file ({file_size} lines) - rules + verification"
            }
        
        # Large files - rules + targeted LLM
        if file_size > 1000:
            return {
                "use_rules": True,
                "use_llm": False,  # Too expensive for full analysis
                "use_llm_verification": True,
                "reason": f"Large file ({file_size} lines) - targeted analysis"
            }
        
        # Default: rules + LLM
        return {
            "use_rules": True,
            "use_llm": True,
            "use_llm_verification": True,
            "reason": "Standard analysis"
        }
    
    def _is_critical_file(self, file_path: str) -> bool:
        """Check if file contains critical functionality"""
        critical_keywords = [
            'auth', 'login', 'password', 'payment', 'billing',
            'security', 'crypto', 'token', 'jwt', 'oauth',
            'permission', 'admin', 'role'
        ]
        path_lower = file_path.lower()
        return any(kw in path_lower for kw in critical_keywords)
    
    def _is_complex_code(self, code: str) -> bool:
        """Detect complex logic that needs LLM"""
        complexity_indicators = [
            r'for\s+.*\s+for\s+',  # Nested loops
            r'if\s+.*\s+if\s+.*\s+if',  # Deep nesting
            r'try\s+.*\s+except\s+.*\s+except',  # Complex error handling
            r'async\s+.*\s+await\s+.*\s+await',  # Complex async
        ]
        import re
        return any(re.search(pattern, code, re.DOTALL) for pattern in complexity_indicators)
    
    def get_cached_analysis(self, code: str, file_path: str) -> Dict | None:
        """Get cached LLM analysis if available"""
        file_hash = self._hash_code(code)
        cache_key = f"{file_path}:{file_hash}"
        return self.cache.get(cache_key)
    
    def cache_analysis(self, code: str, file_path: str, analysis: Dict):
        """Cache LLM analysis results"""
        file_hash = self._hash_code(code)
        cache_key = f"{file_path}:{file_hash}"
        self.cache[cache_key] = {
            "analysis": analysis,
            "timestamp": str(Path.ctime(Path(file_path)) if Path(file_path).exists() else 0)
        }
        self._save_cache()
    
    def _hash_code(self, code: str) -> str:
        """Generate hash of code for caching"""
        return hashlib.md5(code.encode()).hexdigest()[:12]
    
    def _load_cache(self) -> Dict:
        """Load cache from disk"""
        if self.cache_file.exists():
            try:
                with open(self.cache_file) as f:
                    return json.load(f)
            except:
                return {}
        return {}
    
    def _save_cache(self):
        """Save cache to disk"""
        try:
            with open(self.cache_file, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except:
            pass