# Copy ALL your original functions here:
# - RequirementParser
# - AdvancedQualityAnalyzer
# - AdvancedSecurityAnalyzer
# - RequirementComplianceChecker

import json
import re
from pathlib import Path
from enhanced_parser import EnhancedRequirementParser
from smart_analyzer import SmartAnalyzer

# Check for optional dependencies
try:
    import fitz
    PYMUPDF_AVAILABLE = True
except ImportError:
    PYMUPDF_AVAILABLE = False

FALSE_POSITIVE_PATTERNS = {
    'ignore_files': [r'test_', r'_test\.py$', r'\.min\.js$', r'dist/', r'build/', r'node_modules/', r'\.designer\.cs$',
        r'\.g\.cs$',
        r'AssemblyInfo\.cs$',
        r'TemporaryGeneratedFile_.*\.cs$',
        r'\.cshtml\.g\.cs$',
        r'obj/',
        r'bin/',
        r'packages/',
        r'\.nuget/'],
    'framework_files': [r'migrations/', r'vendor/', r'node_modules/', r'\.venv/', r'venv/',   r'Program\.cs$',
        r'Startup\.cs$',
        r'Global\.asax\.cs$',
        r'WebApiConfig\.cs$',
        r'RouteConfig\.cs$',
        r'FilterConfig\.cs$',
        r'BundleConfig\.cs$']
}

class RequirementParser:
    """Parse requirements from PDF, text, or JSON documents."""
    
    def parse_json(self, json_content: str) -> dict:
        """
        Parse JSON format requirements.
        Supports both single requirement object and array of requirements.
        
        Expected format:
        {
            "id": "FR-01",
            "title": "User Login",
            "description": "...",
            "acceptance_criteria": ["...", "..."],
            "nfrs": ["...", "..."]
        }
        """
        try:
            data = json.loads(json_content)
        except json.JSONDecodeError as e:
            return {"error": f"Invalid JSON: {e}", "requirements": [], "validation": {}}
        
        # Handle array of requirements or single requirement
        if isinstance(data, list):
            requirements_list = data
        elif isinstance(data, dict):
            requirements_list = [data]
        else:
            return {"error": "JSON must be object or array", "requirements": [], "validation": {}}
        
        all_requirements = []
        validation_results = []
        
        for req_obj in requirements_list:
            # Validate the requirement structure
            validation = self._validate_requirement(req_obj)
            validation_results.append(validation)
            
            # Extract requirements even if validation fails (for partial analysis)
            extracted = self._extract_from_json_object(req_obj)
            all_requirements.extend(extracted)
        
        return {
            "requirements": all_requirements,
            "validation": validation_results,
            "raw_data": requirements_list
        }
    
    def _validate_requirement(self, req: dict) -> dict:
        """Validate requirement structure and completeness."""
        validation = {
            "id": req.get("id", "UNKNOWN"),
            "title": req.get("title", "Untitled"),
            "is_valid": True,
            "issues": [],
            "score": 100
        }
        
        # Check required fields
        if not req.get("id"):
            validation["issues"].append("Missing requirement ID")
            validation["score"] -= 20
        
        if not req.get("title"):
            validation["issues"].append("Missing title")
            validation["score"] -= 10
        
        if not req.get("description"):
            validation["issues"].append("Missing description")
            validation["score"] -= 10
        
        # Check acceptance criteria (CRITICAL)
        acceptance_criteria = req.get("acceptance_criteria", [])
        if not acceptance_criteria or len(acceptance_criteria) == 0:
            validation["issues"].append("❌ CRITICAL: Missing acceptance criteria - requirement is not testable")
            validation["score"] -= 40
            validation["is_valid"] = False
        elif len(acceptance_criteria) < 2:
            validation["issues"].append("⚠️ Only 1 acceptance criterion - consider adding more scenarios")
            validation["score"] -= 10
        
        # Validate acceptance criteria format (Given/When/Then)
        for i, ac in enumerate(acceptance_criteria):
            if not self._is_valid_acceptance_criteria(ac):
                validation["issues"].append(f"⚠️ Criterion {i+1} doesn't follow Given/When/Then format")
                validation["score"] -= 5
        
        # Check NFRs
        nfrs = req.get("nfrs", [])
        if not nfrs or len(nfrs) == 0:
            validation["issues"].append("⚠️ No non-functional requirements specified")
            validation["score"] -= 10
        
        # Check for vague description
        description = req.get("description", "")
        if description and len(description) < 30:
            validation["issues"].append("⚠️ Description is too brief")
            validation["score"] -= 5
        
        if "easily" in description.lower() or "should be able to" in description.lower():
            if "within" not in description.lower() and "second" not in description.lower():
                validation["issues"].append("⚠️ Description uses vague terms without measurable criteria")
                validation["score"] -= 10
        
        validation["score"] = max(0, validation["score"])
        validation["is_valid"] = validation["score"] >= 60 and len(acceptance_criteria) > 0
        
        return validation
    
    def _is_valid_acceptance_criteria(self, criteria: str) -> bool:
        """Check if acceptance criteria follows Given/When/Then format."""
        criteria_lower = criteria.lower()
        has_given = "given" in criteria_lower
        has_when = "when" in criteria_lower
        has_then = "then" in criteria_lower
        
        return (has_given and has_when and has_then) or \
               (has_when and has_then) or \
               len(criteria) > 50  # Allow longer descriptive criteria
    
    def _extract_from_json_object(self, req: dict) -> list:
        """Extract testable requirements from JSON object."""
        requirements = []
        req_id = req.get("id", "REQ-?")
        title = req.get("title", "")
        
        # Main requirement from description
        if req.get("description"):
            requirements.append({
                "id": req_id,
                "text": f"{title}: {req.get('description')}",
                "type": "functional",
                "source": "description"
            })
        
        # Each acceptance criteria becomes a requirement
        for i, ac in enumerate(req.get("acceptance_criteria", [])):
            requirements.append({
                "id": f"{req_id}-AC{i+1}",
                "text": ac,
                "type": "acceptance",
                "source": "acceptance_criteria",
                "parent_id": req_id
            })
        
        # Each NFR becomes a requirement
        for i, nfr in enumerate(req.get("nfrs", [])):
            # Determine NFR type
            nfr_lower = nfr.lower()
            if "security" in nfr_lower or "password" in nfr_lower or "https" in nfr_lower:
                nfr_type = "security"
            elif "performance" in nfr_lower or "second" in nfr_lower or "latency" in nfr_lower:
                nfr_type = "performance"
            elif "usability" in nfr_lower or "user" in nfr_lower:
                nfr_type = "usability"
            elif "availability" in nfr_lower or "uptime" in nfr_lower:
                nfr_type = "availability"
            else:
                nfr_type = "non-functional"
            
            requirements.append({
                "id": f"{req_id}-NFR{i+1}",
                "text": nfr,
                "type": nfr_type,
                "source": "nfr",
                "parent_id": req_id
            })
        
        return requirements
    
    def parse_text(self, text: str) -> dict:
        """Parse plain text or try JSON first."""
        text = text.strip()
        
        # Try JSON first
        if text.startswith('{') or text.startswith('['):
            return self.parse_json(text)
        
        # Fall back to plain text parsing
        requirements = self._extract_text_requirements(text)
        return {
            "requirements": requirements,
            "validation": [],
            "raw_data": None
        }
    
    def _extract_text_requirements(self, text: str) -> list:
        """Extract requirements from plain text."""
        requirements = []
        seen = set()
        lines = text.split('\n')
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 10:
                continue
            
            # Check for FR-XX: format
            req_match = re.match(r'^(?:FR|REQ|R|NR|NFR|SR|AC)[-\s]?(\d+)[:\s.\-]+(.+)$', line, re.IGNORECASE)
            if req_match:
                req_id, req_text = req_match.groups()
                if req_text.lower() not in seen:
                    requirements.append({
                        'id': f"REQ-{req_id}",
                        'text': req_text.strip(),
                        'type': self._classify_requirement(req_text),
                        'source': 'text'
                    })
                    seen.add(req_text.lower())
        
        return requirements
    
    def _classify_requirement(self, text: str) -> str:
        text_lower = text.lower()
        if any(kw in text_lower for kw in ['auth', 'login', 'password', 'token', 'jwt', 'encrypt', 'security', 'hash', 'https']):
            return 'security'
        elif any(kw in text_lower for kw in ['api', 'endpoint', 'route', 'request', 'response']):
            return 'api'
        elif any(kw in text_lower for kw in ['test', 'coverage']):
            return 'testing'
        elif any(kw in text_lower for kw in ['performance', 'speed', 'latency', 'second', 'millisecond']):
            return 'performance'
        else:
            return 'functional'
    
    def parse_pdf(self, pdf_bytes: bytes) -> dict:
        """Extract requirements from PDF."""
        if PYMUPDF_AVAILABLE:
            try:
                doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                text = ""
                for page in doc:
                    text += page.get_text()
                return self.parse_text(text)
            except Exception as e:
                return {"error": str(e), "requirements": [], "validation": []}
        else:
            return {"error": "PyMuPDF not installed", "requirements": [], "validation": []}


# ============================================
# Multi-Language Analyzers
# ============================================
class AdvancedQualityAnalyzer:
    """Quality analyzer with false positive reduction."""
    
    LANGUAGE_EXTENSIONS = {
        '.py': 'python', '.js': 'javascript', '.jsx': 'javascript',
        '.ts': 'typescript', '.tsx': 'typescript', '.java': 'java',
        '.go': 'go', '.cs': 'csharp', '.cpp': 'cpp','.csproj': 'csharp',  # NEW
    '.vb': 'vb.net',      # NEW
    '.aspx': 'aspnet',    # NEW
    '.cshtml': 'razor',   # NEW
    '.razor': 'blazor', '.c': 'c',
    }
    
    def get_language(self, file_path: str) -> str:
        ext = Path(file_path).suffix.lower()
        return self.LANGUAGE_EXTENSIONS.get(ext, 'unknown')
    
    def should_ignore_file(self, file_path: str) -> bool:
        for pattern in FALSE_POSITIVE_PATTERNS['ignore_files']:
            if re.search(pattern, file_path, re.IGNORECASE):
                return True
        return False
    
    def is_framework_file(self, file_path: str) -> bool:
        for pattern in FALSE_POSITIVE_PATTERNS['framework_files']:
            if re.search(pattern, file_path, re.IGNORECASE):
                return True
        return False
    
    def analyze(self, code: str, file_path: str) -> list:
        if self.should_ignore_file(file_path):
            return []
        
        issues = []
        lang = self.get_language(file_path)
        lines = code.split('\n')
        is_framework = self.is_framework_file(file_path)
        
        issues.extend(self._check_complexity(code, lines, lang, is_framework))
        issues.extend(self._check_naming(code, lang))
        issues.extend(self._check_code_smells(lines, lang, is_framework))
        
        return issues
    
    def _check_complexity(self, code, lines, lang, is_framework):
        issues = []
        if is_framework:
            return issues
        
        func_patterns = {
            'python': r'def\s+(\w+)\s*\(',
            'javascript': r'(?:function\s+(\w+)|(?:const|let|var)\s+(\w+)\s*=\s*(?:async\s+)?(?:function|\([^)]*\)\s*=>))',
            'java': r'(?:public|private|protected)\s+\w+\s+(\w+)\s*\(',
            'go': r'func\s+(?:\([^)]+\)\s+)?(\w+)\s*\(',
            'csharp': r'(?:public|private|protected)\s+\w+\s+(\w+)\s*\(',
            'csharp': r'(?:public|private|protected|internal)\s+(?:static\s+)?(?:async\s+)?(?:virtual\s+)?(?:override\s+)?\w+\s+(\w+)\s*\(',
        }
        
        pattern = func_patterns.get(lang)
        if pattern:
            for match in re.finditer(pattern, code):
                func_name = match.group(1) or 'anonymous'
                start = code[:match.start()].count('\n') + 1
                
                brace_count, func_lines, started = 0, 0, False
                for char in code[match.start():]:
                    if char == '{': brace_count += 1; started = True
                    elif char == '}':
                        brace_count -= 1
                        if started and brace_count == 0: break
                    elif char == '\n': func_lines += 1
                
                if func_lines > 60:
                    issues.append({
                        "category": "Quality", "severity": "Warning", "line": start,
                        "explanation": f"Function '{func_name}' is {func_lines} lines long.",
                        "impact": "Large functions are harder to test and maintain.",
                        "suggested_fix": f"Break '{func_name}' into smaller functions."
                    })
        return issues
    
    def _check_naming(self, code, lang):
        issues = []
        if lang == 'python':
            import ast
            try:
                tree = ast.parse(code)
                for node in ast.walk(tree):
                    if isinstance(node, ast.FunctionDef):
                        if not node.name.startswith('_') and not re.match(r'^[a-z_][a-z0-9_]*$', node.name):
                            issues.append({
                                "category": "Quality", "severity": "Info", "line": node.lineno,
                                "explanation": f"Function '{node.name}' doesn't follow snake_case.",
                                "impact": "Inconsistent naming reduces readability.",
                                "suggested_fix": "Rename to snake_case format."
                            })
                    elif isinstance(node, ast.ClassDef):
                        if not re.match(r'^[A-Z][a-zA-Z0-9]*$', node.name):
                            issues.append({
                                "category": "Quality", "severity": "Info", "line": node.lineno,
                                "explanation": f"Class '{node.name}' doesn't follow PascalCase.",
                                "impact": "Inconsistent naming.",
                                "suggested_fix": "Rename to PascalCase."
                            })
            except SyntaxError as e:
                issues.append({
                    "category": "Quality", "severity": "Error", "line": e.lineno or 1,
                    "explanation": f"Syntax error: {e.msg}",
                    "impact": "Code will not execute.",
                    "suggested_fix": "Fix the syntax error."
                })
        elif lang == 'csharp':  # NEW
        # Check PascalCase for classes
         class_pattern = r'(?:public|private|internal)\s+(?:static\s+)?(?:sealed\s+)?class\s+(\w+)'
         for match in re.finditer(class_pattern, code):
            class_name = match.group(1)
            if not re.match(r'^[A-Z][a-zA-Z0-9]*$', class_name):
                line = code[:match.start()].count('\n') + 1
                issues.append({
                    "category": "Quality",
                    "severity": "Info",
                    "line": line,
                    "explanation": f"Class '{class_name}' doesn't follow PascalCase.",
                    "impact": ".NET convention is PascalCase for classes.",
                    "suggested_fix": "Rename to PascalCase."
                })
        
        # Check private fields naming (_camelCase)
            field_pattern = r'private\s+\w+\s+(\w+)\s*[;=]'
            for match in re.finditer(field_pattern, code):
                field_name = match.group(1)
                if not re.match(r'^_[a-z][a-zA-Z0-9]*$', field_name):
                    line = code[:match.start()].count('\n') + 1
                    issues.append({
                        "category": "Quality",
                        "severity": "Info",
                        "line": line,
                        "explanation": f"Private field '{field_name}' should use _camelCase.",
                        "impact": ".NET convention for private fields.",
                        "suggested_fix": "Rename to _camelCase (e.g., _myField)."
                    })
            
            # Check async methods ending with Async
            async_pattern = r'(?:public|private|protected)\s+async\s+\w+\s+(\w+)\s*\('
            for match in re.finditer(async_pattern, code):
                method_name = match.group(1)
                if not method_name.endswith('Async'):
                    line = code[:match.start()].count('\n') + 1
                    issues.append({
                        "category": "Quality",
                        "severity": "Warning",
                        "line": line,
                        "explanation": f"Async method '{method_name}' should end with 'Async'.",
                        "impact": ".NET convention for async methods.",
                        "suggested_fix": f"Rename to '{method_name}Async'."
                    })
        return issues
    
    def _check_code_smells(self, lines, lang, is_framework):
        issues = []
        for i, line in enumerate(lines, 1):
            if lang == 'python':
                if re.search(r'except\s*:', line) and not is_framework:
                    issues.append({
                        "category": "Quality", "severity": "Warning", "line": i,
                        "explanation": "Bare except clause catches all exceptions.",
                        "impact": "May hide bugs.",
                        "suggested_fix": "Specify exception type."
                    })
            elif lang in ('javascript', 'typescript'):
                if re.search(r'\bvar\s+', line):
                    issues.append({
                        "category": "Quality", "severity": "Info", "line": i,
                        "explanation": "Using 'var' instead of 'let' or 'const'.",
                        "impact": "var has function scope.",
                        "suggested_fix": "Use 'const' or 'let'."
                    })
            if lang == 'csharp':  # NEW
             for i, line in enumerate(lines, 1):
                # Bare catch
                if re.search(r'catch\s*\(\s*Exception\s+\w+\s*\)\s*{\s*}', line):
                    issues.append({
                        "category": "Quality",
                        "severity": "Warning",
                        "line": i,
                        "explanation": "Empty catch block swallows exceptions.",
                        "impact": "Errors are silently ignored.",
                        "suggested_fix": "Log or rethrow the exception."
                    })
                
                # Task.Wait() or Task.Result (blocking async)
                if re.search(r'\.Wait\(\)|\.Result\b', line):
                    issues.append({
                        "category": "Quality",
                        "severity": "Warning",
                        "line": i,
                        "explanation": "Using .Wait() or .Result blocks the thread.",
                        "impact": "Can cause deadlocks in async context.",
                        "suggested_fix": "Use 'await' instead."
                    })
                
                # IDisposable not in using
                if re.search(r'new\s+(?:SqlConnection|StreamReader|HttpClient|FileStream)\(', line):
                    if i < len(lines) and 'using' not in lines[i-1]:
                        issues.append({
                            "category": "Quality",
                            "severity": "Warning",
                            "line": i,
                            "explanation": "IDisposable object not in 'using' statement.",
                            "impact": "Resource leak if exception occurs.",
                            "suggested_fix": "Wrap in 'using' statement."
                        })
        return issues


class AdvancedSecurityAnalyzer:
    """Security analyzer with severity classification."""
    
    def analyze(self, code: str, file_path: str, lang: str) -> list:
        issues = []
        lines = code.split('\n')
        
        issues.extend(self._check_hardcoded_secrets(lines))
        issues.extend(self._check_injection(lines, lang))
        issues.extend(self._check_dangerous_functions(lines, lang))
        issues.extend(self._check_security_best_practices(lines, lang))

        if lang == 'csharp':
            issues.extend(self._check_dotnet_security(lines))
        
        return issues
    
    def _check_hardcoded_secrets(self, lines):
        issues = []
        secret_patterns = [
            (r'(?:password|passwd|pwd)\s*[=:]\s*["\'][^"\']{4,}["\']', 'Hardcoded password'),
            (r'(?:api_?key|apikey|secret_?key)\s*[=:]\s*["\'][^"\']{8,}["\']', 'Hardcoded API key'),
            (r'["\'](?:sk-|pk_live_|sk_live_|ghp_|AKIA)[A-Za-z0-9]{10,}["\']', 'Exposed credential'),
        ]
        
        for i, line in enumerate(lines, 1):
            if line.strip().startswith(('#', '//', '/*', '*')):
                continue
            if any(p in line.lower() for p in ['example', 'placeholder', 'xxx', 'changeme', 'your_', 'todo']):
                continue
            
            for pattern, msg in secret_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append({
                        "category": "Security", "severity": "Error", "line": i,
                        "explanation": f"{msg} detected.",
                        "impact": "Credentials can be leaked via version control.",
                        "suggested_fix": "Use environment variables.",
                        "cwe": "798"
                    })
                    break
        return issues
    
    def _check_injection(self, lines, lang):
        issues = []
        sql_patterns = [
            r'execute\s*\(\s*f["\'].*SELECT',
            r'execute\s*\(\s*["\'].*%s.*%',
            r'query\s*\(\s*["\'].*\+\s*\w+',
        ]
        
        for i, line in enumerate(lines, 1):
            for pattern in sql_patterns:
                if re.search(pattern, line, re.IGNORECASE):
                    issues.append({
                        "category": "Security", "severity": "Error", "line": i,
                        "explanation": "Potential SQL injection.",
                        "impact": "Attackers can execute arbitrary SQL.",
                        "suggested_fix": "Use parameterized queries.",
                        "cwe": "89"
                    })
                    break
        return issues
    
    def _check_dangerous_functions(self, lines, lang):
        issues = []
        dangerous = {
            'python': [
                (r'\beval\s*\(', 'eval() executes arbitrary code', '95'),
                (r'\bexec\s*\(', 'exec() executes arbitrary code', '95'),
                (r'pickle\.loads?\s*\(', 'Pickle deserialization is unsafe', '502'),
            ],
            'javascript': [
                (r'\beval\s*\(', 'eval() executes arbitrary code', '95'),
                (r'innerHTML\s*=\s*[^"\'`]', 'innerHTML can cause XSS', '79'),
            ],
            'csharp': [  # NEW
                (r'Assembly\.Load\(', 'Dynamic assembly loading can execute arbitrary code', '502'),
                (r'\.InnerHtml\s*=', 'Setting InnerHtml can cause XSS', '79'),
                (r'Response\.Write\((?!.*Encode)', 'Response.Write without encoding causes XSS', '79'),
                (r'new\s+X509Certificate2\([^,]+\)', 'Certificate loaded without validation', '295'),
                (r'ServerCertificateValidationCallback\s*=\s*\(', 'SSL certificate validation disabled', '295'),
                (r'ValidateInput\s*\(\s*false\s*\)', 'Input validation disabled', '20'),
                (r'MD5\.Create\(\)|SHA1\.Create\(\)', 'Weak cryptographic algorithm', '327'),
                (r'new\s+(?:DESCryptoServiceProvider|TripleDESCryptoServiceProvider)', 'Weak encryption algorithm', '327'),
            ],

        }
        
        for pattern, msg, cwe in dangerous.get(lang, []):
            for i, line in enumerate(lines, 1):
                if re.search(pattern, line):
                    issues.append({
                        "category": "Security", "severity": "Error", "line": i,
                        "explanation": msg,
                        "impact": "Can lead to code execution.",
                        "suggested_fix": "Use safer alternatives.",
                        "cwe": cwe
                    })
            return issues
        
        def _check_security_best_practices(self, lines, lang):
            """Check security best practices."""
            issues = []
            
            if lang == 'csharp':
                issues.extend(self._check_dotnet_security(lines))
            
            return issues
        
        def _check_dotnet_security(self, lines):
            """Check .NET-specific security issues"""
            issues = []
        
        for i, line in enumerate(lines, 1):
            # SQL injection in Entity Framework
            if re.search(r'\.FromSqlRaw\s*\(\s*\$', line):
                issues.append({
                    "category": "Security",
                    "severity": "Error",
                    "line": i,
                    "explanation": "SQL injection via FromSqlRaw with string interpolation.",
                    "impact": "Allows arbitrary SQL execution.",
                    "suggested_fix": "Use parameterized queries: FromSqlRaw(sql, param1, param2)",
                    "cwe": "89"
                })
            
            # Hardcoded connection string
            if re.search(r'(?:connectionString|ConnectionString)\s*=\s*["\'].*Server=.*Password=', line, re.IGNORECASE):
                if 'example' not in line.lower() and 'placeholder' not in line.lower():
                    issues.append({
                        "category": "Security",
                        "severity": "Error",
                        "line": i,
                        "explanation": "Hardcoded database connection string with credentials.",
                        "impact": "Credentials exposed in source code.",
                        "suggested_fix": "Use configuration file or environment variables.",
                        "cwe": "798"
                    })
            
            # Missing [Authorize] attribute on controller
            if re.search(r'public\s+class\s+\w+Controller\s*:', line):
                # Check if [Authorize] appears in previous 5 lines
                has_authorize = any('[Authorize' in lines[max(0, i-5):i][j] for j in range(len(lines[max(0, i-5):i])))
                if not has_authorize:
                    issues.append({
                        "category": "Security",
                        "severity": "Warning",
                        "line": i,
                        "explanation": "Controller missing [Authorize] attribute.",
                        "impact": "All actions are publicly accessible.",
                        "suggested_fix": "Add [Authorize] attribute to controller or actions.",
                        "cwe": "862"
                    })
        
        return issues


# ============================================
# Requirement Compliance Checker (Enhanced)
# ============================================
class RequirementComplianceChecker:
    """Cross-verify code against requirements with detailed matching."""
    
    def check_compliance(self, code_files: list, requirements: list) -> list:
        """Check if requirements are implemented in code."""
        compliance_results = []
        
        all_code = "\n".join([f.get('code', '') for f in code_files])
        all_code_lower = all_code.lower()
        
        for req in requirements:
            req_text = req.get('text', '')
            req_id = req.get('id', 'REQ-?')
            req_type = req.get('type', 'functional')
            req_source = req.get('source', 'unknown')
            
            # Extract key terms
            key_terms = self._extract_key_terms(req_text)
            
            # Search for implementation
            found_files = []
            found_evidence = []
            match_score = 0
            
            for file_info in code_files:
                file_code = file_info.get('code', '')
                file_code_lower = file_code.lower()
                file_path = file_info.get('path', '')
                
                matches = sum(1 for term in key_terms if term.lower() in file_code_lower)
                if matches > 0:
                    found_files.append(file_path)
                    match_score = max(match_score, matches / len(key_terms) if key_terms else 0)
                    
                    # Find specific evidence
                    for term in key_terms:
                        if term.lower() in file_code_lower:
                            # Find the line containing this term
                            for line_num, line in enumerate(file_code.split('\n'), 1):
                                if term.lower() in line.lower():
                                    found_evidence.append(f"{file_path}:{line_num}")
                                    break
            
            # Special handling for different requirement types
            if req_type == 'security':
                match_score = self._check_security_requirement(req_text, all_code_lower, match_score)
            elif req_type == 'performance':
                # Performance requirements need specific checks
                match_score = self._check_performance_requirement(req_text, all_code_lower, match_score)
            
            # Determine status
            if match_score >= 0.6:
                status = "✅ Fully implemented"
                explanation = f"Found in: {', '.join(found_files[:3])}"
            elif match_score >= 0.3:
                status = "⚠️ Partially implemented"
                explanation = f"Partial match. Some aspects may be missing."
            else:
                status = "❌ Not implemented"
                explanation = f"No implementation found for: {req_text[:60]}..."
            
            compliance_results.append({
                'requirement_id': req_id,
                'requirement': req_text,
                'type': req_type,
                'source': req_source,
                'status': status,
                'explanation': explanation,
                'files': found_files[:5],
                'evidence': found_evidence[:5],
                'score': round(match_score, 2)
            })
        
        return compliance_results
    
    def _extract_key_terms(self, text: str) -> list:
        stop_words = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 
                      'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would',
                      'should', 'shall', 'must', 'can', 'may', 'might', 'to', 'of',
                      'in', 'for', 'on', 'with', 'at', 'by', 'from', 'as', 'into',
                      'and', 'or', 'but', 'if', 'then', 'else', 'when', 'where',
                      'that', 'this', 'these', 'those', 'it', 'its', 'given', 'user'}
        
        words = re.findall(r'\b[a-zA-Z_][a-zA-Z0-9_]*\b', text.lower())
        terms = [w for w in words if w not in stop_words and len(w) > 2]
        return terms[:15]
    
    def _check_security_requirement(self, req_text: str, code: str, base_score: float) -> float:
        """Enhanced checking for security requirements."""
        req_lower = req_text.lower()
        
        # Check for HTTPS
        if 'https' in req_lower:
            if 'https' in code or 'ssl' in code or 'tls' in code:
                base_score = max(base_score, 0.7)
        
        # Check for hashing
        if 'hash' in req_lower or 'hashing' in req_lower:
            if any(h in code for h in ['bcrypt', 'argon2', 'pbkdf2', 'hashlib', 'crypto']):
                base_score = max(base_score, 0.8)
        
        # Check for password security
        if 'password' in req_lower:
            if 'hash' in code or 'encrypt' in code or 'bcrypt' in code:
                base_score = max(base_score, 0.7)
        
        return base_score
    
    def _check_performance_requirement(self, req_text: str, code: str, base_score: float) -> float:
        """Check for performance-related implementations."""
        req_lower = req_text.lower()
        
        # Check for timeout/timing implementations
        if 'second' in req_lower or 'millisecond' in req_lower:
            if any(t in code for t in ['timeout', 'async', 'await', 'promise', 'cache']):
                base_score = max(base_score, 0.6)
        
        return base_score
    
    def calculate_score(self, compliance_results: list) -> dict:
        """Calculate compliance score."""
        total = len(compliance_results)
        if total == 0:
            return {'score': 100, 'result': 'PASS', 'fully': 0, 'partial': 0, 'none': 0, 'total': 0}
        
        fully = sum(1 for r in compliance_results if '✅' in r['status'])
        partial = sum(1 for r in compliance_results if '⚠️' in r['status'])
        none = sum(1 for r in compliance_results if '❌' in r['status'])
        
        earned = fully * 1.0 + partial * 0.5 + none * 0.0
        score = (earned / total) * 100
        
        return {
            'total': total,
            'fully': fully,
            'partial': partial,
            'none': none,
            'score': round(score, 1),
            'result': 'PASS' if score >= 75 else 'FAIL'
        }


# Then add this wrapper at the bottom:
class AgentToolbox:
    def __init__(self):
        self.parser = RequirementParser()
        self.enhanced_parser = EnhancedRequirementParser()  # NEW
        self.quality = AdvancedQualityAnalyzer()
        self.security = AdvancedSecurityAnalyzer()
        self.compliance = RequirementComplianceChecker()
        self.smart_analyzer = SmartAnalyzer()  # NEW
        try:
            from llm_analyzer import LLMCodeAnalyzer
            self.llm = LLMCodeAnalyzer()
        except ImportError:
            self.llm = None