"""
Flask API for Code Review Agent
Handles: direct code paste, file upload, GitHub repo analysis
"""

from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import tempfile
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List

# Import your existing modules
from agent_tools import AgentToolbox
from enhanced_parser import EnhancedRequirementParser
from smart_analyzer import SmartAnalyzer

# Optional: OpenAI for agent
try:
    from openai import AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# Optional: LLM analyzer
try:
    from llm_analyzer import LLMCodeAnalyzer
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

app = Flask(__name__)
CORS(app)  # Enable CORS for React frontend

# Initialize tools
toolbox = AgentToolbox()
req_parser = EnhancedRequirementParser()
smart_analyzer = SmartAnalyzer()

if LLM_AVAILABLE:
    llm_analyzer = LLMCodeAnalyzer()
else:
    llm_analyzer = None


class AzureConfig:
    def __init__(self):
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
        self.api_version = "2024-02-15-preview"
    
    def validate(self) -> bool:
        return bool(self.api_key and self.endpoint and self.deployment)


@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    return jsonify({
        "status": "healthy",
        "llm_available": LLM_AVAILABLE and (llm_analyzer.ollama_available if llm_analyzer else False),
        "openai_available": OPENAI_AVAILABLE
    })


@app.route('/analyze', methods=['POST'])
def analyze_code():
    """
    Main analysis endpoint
    Accepts:
    - direct_code: string (code pasted directly)
    - uploaded_files: array of {filename, content}
    - github_url: string (GitHub repo URL)
    - requirements: string (any text format)
    - ai_mode: "off"|"smart"|"always"
    """
    
    try:
        data = request.json
        
        # Extract inputs
        direct_code = data.get('direct_code')
        uploaded_files = data.get('uploaded_files', [])
        github_url = data.get('github_url')
        requirements_text = data.get('requirements', '')
        ai_mode = data.get('ai_mode', 'smart')
        
        # Validate inputs
        if not any([direct_code, uploaded_files, github_url]):
            return jsonify({"error": "No code provided"}), 400
        
        # Parse requirements
        requirements = []
        if requirements_text.strip():
            parsed = req_parser.parse(requirements_text)
            requirements = parsed.get("requirements", [])
        
        # Get code files
        code_files = []
        
        # 1. Direct code paste
        if direct_code:
            language = detect_language(direct_code)
            code_files.append({
                "path": f"main.{get_extension(language)}",
                "code": direct_code,
                "language": language
            })
        
        # 2. Uploaded files
        if uploaded_files:
            for file_data in uploaded_files:
                filename = file_data.get('filename', 'unknown.txt')
                content = file_data.get('content', '')
                ext = filename.split('.')[-1]
                
                lang_map = {
                    'py': 'python', 'js': 'javascript', 'ts': 'typescript',
                    'java': 'java', 'cs': 'csharp', 'go': 'go',
                    'cpp': 'cpp', 'c': 'c'
                }
                
                code_files.append({
                    "path": filename,
                    "code": content,
                    "language": lang_map.get(ext, 'unknown')
                })
        
        # 3. GitHub repository
        if github_url:
            repo_files = clone_and_extract_files(github_url)
            code_files.extend(repo_files)
        
        if not code_files:
            return jsonify({"error": "No valid code files found"}), 400
        
        # Analyze each file
        all_results = []
        
        for file_info in code_files:
            file_result = analyze_single_file(
                file_info['code'],
                file_info['path'],
                file_info['language'],
                requirements,
                ai_mode
            )
            all_results.append(file_result)
        
        # Generate final report
        final_report = generate_final_report(all_results, requirements)
        
        return jsonify(final_report)
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500


def analyze_single_file(
    code: str,
    file_path: str,
    language: str,
    requirements: List[Dict],
    ai_mode: str
) -> Dict:
    """Analyze a single code file"""
    
    # Step 1: Determine what to check
    analysis_plan = smart_analyzer.analyze_code_type(code, file_path, language)
    
    quality_issues = []
    security_issues = []
    performance_issues = []
    semantic_issues = []
    compliance_results = []
    
    # Step 2: Run quality analysis (always)
    if analysis_plan["checks_needed"]["quality"]:
        quality_issues = toolbox.quality.analyze(code, file_path)
    
    # Step 3: Run security analysis (if needed)
    if analysis_plan["checks_needed"]["security"]:
        security_issues = toolbox.security.analyze(code, file_path, language)
    
    # Step 4: Run LLM semantic analysis (if AI enabled and available)
    if ai_mode != "off" and llm_analyzer and llm_analyzer.ollama_available:
        if analysis_plan["checks_needed"]["performance"] or analysis_plan["code_type"] == "dsa":
            llm_result = llm_analyzer.analyze_code_semantics(code, file_path, language)
            semantic_issues = llm_result.get("semantic_issues", [])
    
    # Step 5: Check requirements compliance
    if requirements and analysis_plan["checks_needed"]["requirements"]:
        compliance_results = toolbox.compliance.check_compliance(
            [{"path": file_path, "code": code}],
            requirements
        )
    
    # Step 6: Generate PASS/FAIL report
    report = smart_analyzer.generate_pass_fail_report(
        code_type=analysis_plan["code_type"],
        checks_run=analysis_plan["checks_needed"],
        quality_issues=quality_issues,
        security_issues=security_issues,
        performance_issues=semantic_issues,  # LLM finds performance issues
        compliance_results=compliance_results
    )
    
    return {
        "file": file_path,
        "language": language,
        "code_type": analysis_plan["code_type"],
        "reasoning": analysis_plan["reasoning"],
        "report": report,
        "issues": {
            "quality": quality_issues,
            "security": security_issues,
            "semantic": semantic_issues
        }
    }


def generate_final_report(file_results: List[Dict], requirements: List[Dict]) -> Dict:
    """Generate final aggregated report"""
    
    total_files = len(file_results)
    passed_files = sum(1 for r in file_results if r["report"]["overall_result"] == "PASS")
    failed_files = total_files - passed_files
    
    avg_score = sum(r["report"]["score"] for r in file_results) / total_files if total_files > 0 else 0
    
    overall_pass = passed_files >= (total_files * 0.7)  # 70% of files must pass
    
    return {
        "overall_result": "PASS" if overall_pass else "FAIL",
        "overall_score": round(avg_score, 1),
        "summary": {
            "total_files": total_files,
            "passed": passed_files,
            "failed": failed_files,
            "requirements_checked": len(requirements)
        },
        "files": file_results,
        "message": f"{'✅ All checks passed!' if overall_pass else '❌ Some files failed review'}"
    }


def detect_language(code: str) -> str:
    """Detect programming language from code"""
    code_lower = code.lower()
    
    if 'def ' in code or 'import ' in code or 'print(' in code:
        return 'python'
    elif 'function ' in code or 'const ' in code or 'let ' in code or '=>' in code:
        return 'javascript'
    elif 'public class' in code or 'public static void main' in code:
        return 'java'
    elif 'using system' in code_lower or 'namespace ' in code_lower:
        return 'csharp'
    elif 'package main' in code or 'func main()' in code:
        return 'go'
    else:
        return 'unknown'


def get_extension(language: str) -> str:
    """Get file extension for language"""
    ext_map = {
        'python': 'py',
        'javascript': 'js',
        'typescript': 'ts',
        'java': 'java',
        'csharp': 'cs',
        'go': 'go',
        'cpp': 'cpp',
        'c': 'c'
    }
    return ext_map.get(language, 'txt')


def clone_and_extract_files(github_url: str) -> List[Dict]:
    """Clone GitHub repo and extract code files"""
    
    # Create temp directory
    temp_dir = tempfile.mkdtemp()
    
    try:
        # Clone repo
        subprocess.run(
            ['git', 'clone', '--depth', '1', github_url, temp_dir],
            check=True,
            capture_output=True
        )
        
        # Extract code files
        code_files = []
        extensions = {'.py', '.js', '.ts', '.java', '.cs', '.go', '.cpp', '.c'}
        
        for root, dirs, files in os.walk(temp_dir):
            # Skip common non-code directories
            dirs[:] = [d for d in dirs if d not in {'.git', 'node_modules', 'venv', '__pycache__', 'dist', 'build'}]
            
            for file in files:
                file_path = Path(root) / file
                if file_path.suffix in extensions:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                        
                        relative_path = file_path.relative_to(temp_dir)
                        
                        lang_map = {
                            '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
                            '.java': 'java', '.cs': 'csharp', '.go': 'go',
                            '.cpp': 'cpp', '.c': 'c'
                        }
                        
                        code_files.append({
                            "path": str(relative_path),
                            "code": content,
                            "language": lang_map.get(file_path.suffix, 'unknown')
                        })
                    except:
                        continue
        
        return code_files[:20]  # Limit to 20 files
    
    except subprocess.CalledProcessError:
        raise Exception("Failed to clone GitHub repository")
    
    finally:
        # Cleanup
        shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == '__main__':
    port = int(os.getenv('PORT', 5000))
    app.run(debug=True, host='0.0.0.0', port=port)