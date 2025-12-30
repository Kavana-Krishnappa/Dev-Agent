"""
Smart Analyzer - Decides what to check based on code type
"""

from typing import Dict, List

class SmartAnalyzer:
    
    def analyze_code_type(self, code: str, file_path: str, language: str) -> Dict:
        code_lower = code.lower()
        code_type = self._detect_code_type(code, code_lower, file_path)
        checks_needed = self._decide_checks(code_type)
        reasoning = self._generate_reasoning(code_type)
        
        return {
            "code_type": code_type,
            "checks_needed": checks_needed,
            "reasoning": reasoning
        }
    
    def _detect_code_type(self, code: str, code_lower: str, file_path: str) -> str:
        # DSA indicators
        dsa_keywords = ['def sort', 'def search', 'binary search', 'def dfs', 'def bfs', 
                       'class node', 'treenode', 'listnode', 'dp[', 'memo[', 'visited[']
        dsa_count = sum(1 for kw in dsa_keywords if kw in code_lower)
        
        # Web app indicators
        web_keywords = ['app.route', '@app.', 'flask', 'fastapi', 'django', 'express',
                       'router', 'middleware', 'session', 'cookie', 'jwt', 'oauth']
        web_count = sum(1 for kw in web_keywords if kw in code_lower)
        
        # Security indicators
        security_keywords = ['password', 'hash', 'encrypt', 'auth', 'login', 'token', 'jwt']
        security_count = sum(1 for kw in security_keywords if kw in code_lower)
        
        # Database indicators
        db_keywords = ['sql', 'database', 'query', 'select ', 'insert ', 'connection', 'execute']
        db_count = sum(1 for kw in db_keywords if kw in code_lower)
        
        # Decision
        if dsa_count >= 2 and web_count == 0:
            return "dsa"
        if web_count >= 2:
            return "web_app"
        if db_count >= 3:
            return "database"
        if security_count >= 2:
            return "security_critical"
        
        return "general"
    
    def _decide_checks(self, code_type: str) -> Dict[str, bool]:
        checks = {
            "quality": True,
            "security": False,
            "performance": False,
            "requirements": True
        }
        
        if code_type == "dsa":
            checks["performance"] = True
            checks["security"] = False
        elif code_type in ["web_app", "database", "security_critical"]:
            checks["quality"] = True
            checks["security"] = True
            checks["performance"] = True
            checks["requirements"] = True
        else:
            checks["quality"] = True
            checks["security"] = False
            checks["requirements"] = True
        
        return checks
    
    def _generate_reasoning(self, code_type: str) -> str:
        reasons = {
            "dsa": "Algorithm/DSA code detected. Checking quality and performance, skipping security (not relevant).",
            "web_app": "Web application detected. Running all checks including security (XSS, CSRF, auth).",
            "database": "Database code detected. Checking for SQL injection and security issues.",
            "security_critical": "Security-sensitive code detected. Running comprehensive security analysis.",
            "general": "General code. Running quality and requirement checks."
        }
        return reasons.get(code_type, "Running standard analysis.")
    
    def generate_pass_fail_report(self, code_type: str, checks_run: Dict,
                                  quality_issues: List, security_issues: List,
                                  performance_issues: List, compliance_results: List) -> Dict:
        
        checks_report = {}
        total_score = 0
        max_score = 0
        
        # Quality
        if checks_run.get("quality"):
            critical = sum(1 for i in quality_issues if i.get("severity") == "Error")
            warnings = sum(1 for i in quality_issues if i.get("severity") == "Warning")
            
            if critical == 0 and warnings <= 3:
                status = "PASS"
                reason = f"Good code quality. {warnings} minor warnings."
                score = 100
            elif critical == 0:
                status = "PASS"
                reason = f"No critical issues. {warnings} warnings found."
                score = 80
            else:
                status = "FAIL"
                reason = f"{critical} critical quality issues."
                score = 40
            
            checks_report["quality"] = {
                "status": status,
                "issues": len(quality_issues),
                "critical": critical,
                "warnings": warnings,
                "reason": reason
            }
            total_score += score
            max_score += 100
        
        # Security
        if checks_run.get("security"):
            critical = sum(1 for i in security_issues if i.get("severity") == "Error")
            
            if critical == 0:
                status = "PASS"
                reason = "No security vulnerabilities detected."
                score = 100
            else:
                status = "FAIL"
                reason = f"{critical} security vulnerabilities (SQL injection, XSS, hardcoded secrets)."
                score = 0
            
            checks_report["security"] = {
                "status": status,
                "issues": len(security_issues),
                "critical": critical,
                "reason": reason
            }
            total_score += score
            max_score += 100
        else:
            checks_report["security"] = {
                "status": "SKIPPED",
                "issues": 0,
                "reason": f"Security checks not required for {code_type} code."
            }
        
        # Performance
        if checks_run.get("performance"):
            perf_count = len(performance_issues)
            if perf_count == 0:
                status = "PASS"
                reason = "No performance issues detected."
                score = 100
            elif perf_count <= 2:
                status = "PASS"
                reason = f"{perf_count} minor performance suggestions."
                score = 80
            else:
                status = "FAIL"
                reason = f"{perf_count} performance issues."
                score = 50
            
            checks_report["performance"] = {
                "status": status,
                "issues": perf_count,
                "reason": reason
            }
            total_score += score
            max_score += 100
        else:
            checks_report["performance"] = {
                "status": "SKIPPED",
                "issues": 0,
                "reason": f"Performance checks not critical for {code_type} code."
            }
        
        # Requirements
        if checks_run.get("requirements") and compliance_results:
            fully = sum(1 for r in compliance_results if "✅" in r.get("status", ""))
            partial = sum(1 for r in compliance_results if "⚠️" in r.get("status", ""))
            missing = sum(1 for r in compliance_results if "❌" in r.get("status", ""))
            total = len(compliance_results)
            
            comp_score = (fully + partial * 0.5) / total * 100 if total > 0 else 100
            
            if comp_score >= 80:
                status = "PASS"
                reason = f"{fully}/{total} requirements fully implemented."
            elif comp_score >= 60:
                status = "PASS"
                reason = f"{fully} fully, {partial} partially implemented."
            else:
                status = "FAIL"
                reason = f"{missing} requirements not implemented."
            
            checks_report["requirements"] = {
                "status": status,
                "fully": fully,
                "partial": partial,
                "missing": missing,
                "total": total,
                "reason": reason
            }
            total_score += comp_score
            max_score += 100
        else:
            checks_report["requirements"] = {
                "status": "SKIPPED",
                "reason": "No requirements provided."
            }
        
        # Overall
        final_score = (total_score / max_score * 100) if max_score > 0 else 100
        overall = "PASS" if final_score >= 70 else "FAIL"
        
        passed = [k for k, v in checks_report.items() if v.get("status") == "PASS"]
        failed = [k for k, v in checks_report.items() if v.get("status") == "FAIL"]
        
        if overall == "PASS":
            summary = f"✅ Code review PASSED ({final_score:.1f}%)\n"
            summary += f"Code type: {code_type}\n"
            summary += f"Passed: {', '.join(passed) if passed else 'None'}"
        else:
            summary = f"❌ Code review FAILED ({final_score:.1f}%)\n"
            summary += f"Code type: {code_type}\n"
            summary += f"Failed: {', '.join(failed) if failed else 'None'}"
        
        return {
            "overall_result": overall,
            "score": round(final_score, 1),
            "checks": checks_report,
            "summary": summary,
            "code_type": code_type
        }