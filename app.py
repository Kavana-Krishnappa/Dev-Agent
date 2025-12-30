"""
COMPLETE FIXED CODE REVIEW AGENT
Smart analysis, flexible requirements, PASS/FAIL reporting + Fixed GitHub Integration
"""

import os
import json
import re
from pathlib import Path
from typing import List, Dict, Optional
import streamlit as st
import requests
from io import BytesIO
import zipfile

# Load env
try:
    from dotenv import load_dotenv
    load_dotenv()
except:
    pass

# Check dependencies
try:
    from openai import AzureOpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

try:
    from llm_analyzer import LLMCodeAnalyzer
    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

from agent_tools import AgentToolbox
from enhanced_parser import EnhancedRequirementParser
from smart_analyzer import SmartAnalyzer

# ============================================
# FIXED GITHUB REPOSITORY FETCHER
# ============================================
class GitHubRepoFetcher:
    """Fetches GitHub repository contents with smart filtering - FIXED VERSION"""
    
    # ONLY ignore truly useless files/folders
    IGNORE_PATTERNS = {
        # Dependency folders (these are huge and useless)
        'node_modules', 'venv', '.venv', 'env', '__pycache__', 
        '.pytest_cache', '.mypy_cache', '.tox', 'vendor',
        'bower_components', 'dist', 'build', 'target', 'out',
        
        # Lock files (too large, not useful for review)
        'package-lock.json', 'yarn.lock', 'poetry.lock', 
        'Pipfile.lock', 'pnpm-lock.yaml', 'composer.lock',
        
        # IDE configs (not code)
        '.vscode', '.idea', '.vs', '.DS_Store',
        
        # Git folder
        '.git',
        
        # Logs
        '*.log', 'logs',
        
        # Coverage reports
        'coverage', 'htmlcov', '.nyc_output',
    }
    
    # Binary/media extensions to skip (can't review these)
    BINARY_EXTENSIONS = {
        '.png', '.jpg', '.jpeg', '.gif', '.ico', '.svg', '.webp',
        '.mp4', '.avi', '.mov', '.mp3', '.wav',
        '.zip', '.tar', '.gz', '.rar', '.7z',
        '.pdf', '.doc', '.docx', '.xls', '.xlsx',
        '.woff', '.woff2', '.ttf', '.eot',
        '.exe', '.dll', '.so', '.dylib',
        '.pyc', '.pyo', '.class', '.o',
        '.min.js', '.min.css',  # Minified files
    }
    
    # Code extensions we CAN review
    CODE_EXTENSIONS = {
        '.py': 'python', '.js': 'javascript', '.ts': 'typescript',
        '.jsx': 'javascript', '.tsx': 'typescript',
        '.java': 'java', '.go': 'go', '.rs': 'rust',
        '.cpp': 'cpp', '.c': 'c', '.h': 'c', '.hpp': 'cpp',
        '.cs': 'csharp', '.rb': 'ruby', '.php': 'php',
        '.kt': 'kotlin', '.swift': 'swift', '.scala': 'scala',
        '.html': 'html', '.css': 'css', '.scss': 'scss',
        '.vue': 'vue', '.sql': 'sql', '.sh': 'bash',
        '.yaml': 'yaml', '.yml': 'yaml', '.json': 'json',
        '.xml': 'xml', '.md': 'markdown', '.txt': 'text',
    }
    
    # Priority patterns (for sorting, NOT filtering)
    PRIORITY_PATTERNS = {
        'critical': [
            r'^README\.md$', r'^ARCHITECTURE\.md$',
            r'^main\.(py|js|ts|java|go|rs|cpp)$',
            r'^index\.(py|js|ts|jsx|tsx|html)$',
            r'^app\.(py|js|ts|jsx|tsx)$',
        ],
        'high': [
            r'^src/.+\.(py|js|ts|jsx|tsx|java|go|rs|cpp|c|h)$',
            r'^app/.+\.(py|js|ts|jsx|tsx)$',
            r'^lib/.+\.(py|js|ts|jsx|tsx)$',
            r'^api/.+\.(py|js|ts)$',
            r'^controllers?/.+\.(py|js|ts)$',
            r'^services?/.+\.(py|js|ts)$',
            r'^models?/.+\.(py|js|ts)$',
        ],
        'medium': [
            r'^tests?/.+',
            r'^config/.+',
            r'^docs?/.+',
        ],
    }
    
    MAX_FILE_SIZE = 500 * 1024  # 500KB per file
    
    def __init__(self, github_token: Optional[str] = None):
        self.github_token = github_token or os.getenv("GITHUB_TOKEN")
        self.session = requests.Session()
        if self.github_token:
            self.session.headers.update({'Authorization': f'token {self.github_token}'})
    
    def parse_github_url(self, url: str) -> Dict[str, str]:
        """Parse GitHub URL to extract owner and repo"""
        patterns = [
            r'github\.com/([^/]+)/([^/]+?)(?:\.git)?/?$',
            r'github\.com/([^/]+)/([^/]+)/tree/([^/]+)/?(.*)$',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, url)
            if match:
                owner, repo = match.group(1), match.group(2)
                branch = match.group(3) if len(match.groups()) >= 3 else 'main'
                path = match.group(4) if len(match.groups()) >= 4 else ''
                return {
                    'owner': owner,
                    'repo': repo,
                    'branch': branch,
                    'path': path
                }
        
        raise ValueError(f"Invalid GitHub URL: {url}")
    
    def should_ignore(self, file_path: str) -> bool:
        """Check if file should be ignored (dependencies, binaries, etc.)"""
        path_parts = Path(file_path).parts
        filename = Path(file_path).name
        
        # Check if any part of the path is in ignore list
        for part in path_parts:
            if part in self.IGNORE_PATTERNS:
                return True
        
        # Check filename patterns
        for pattern in self.IGNORE_PATTERNS:
            if '*' in pattern:
                import fnmatch
                if fnmatch.fnmatch(filename, pattern):
                    return True
        
        # Check binary extensions
        ext = Path(file_path).suffix.lower()
        if ext in self.BINARY_EXTENSIONS:
            return True
        
        return False
    
    def get_priority(self, file_path: str) -> str:
        """Get priority level for sorting (NOT filtering)"""
        for priority, patterns in self.PRIORITY_PATTERNS.items():
            for pattern in patterns:
                if re.search(pattern, file_path, re.IGNORECASE):
                    return priority
        return 'low'  # Changed from 'ignore' to 'low' - include everything!
    
    def is_code_file(self, file_path: str) -> bool:
        """Check if file is a code file we can analyze"""
        ext = Path(file_path).suffix.lower()
        
        # Accept known code extensions
        if ext in self.CODE_EXTENSIONS:
            return True
        
        # Accept files without extension (could be scripts)
        if not ext and Path(file_path).name.lower() in ['makefile', 'dockerfile', 'rakefile']:
            return True
        
        return False
    
    def fetch_repo_zip(self, owner: str, repo: str, branch: str = 'main') -> bytes:
        """Download repository as ZIP"""
        url = f"https://github.com/{owner}/{repo}/archive/refs/heads/{branch}.zip"
        
        try:
            response = self.session.get(url, timeout=30)
            response.raise_for_status()
            return response.content
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                # Try 'master' branch
                url = f"https://github.com/{owner}/{repo}/archive/refs/heads/master.zip"
                response = self.session.get(url, timeout=30)
                response.raise_for_status()
                return response.content
            raise
    
    def extract_files(self, zip_content: bytes) -> tuple:
        """Extract ALL valid code files from ZIP"""
        files = []
        skipped = {'ignored': 0, 'binary': 0, 'too_large': 0, 'decode_error': 0}
        
        with zipfile.ZipFile(BytesIO(zip_content)) as zf:
            for info in zf.infolist():
                # Skip directories
                if info.is_dir():
                    continue
                
                # Get relative path (remove root folder from ZIP)
                parts = Path(info.filename).parts
                if len(parts) <= 1:
                    continue
                rel_path = str(Path(*parts[1:]))
                
                # Check if should ignore (dependencies, IDE files, etc.)
                if self.should_ignore(rel_path):
                    skipped['ignored'] += 1
                    continue
                
                # Check if it's a code file
                if not self.is_code_file(rel_path):
                    skipped['binary'] += 1
                    continue
                
                # Check file size
                if info.file_size > self.MAX_FILE_SIZE:
                    skipped['too_large'] += 1
                    continue
                
                # Try to extract content
                try:
                    content = zf.read(info.filename).decode('utf-8', errors='ignore')
                except Exception:
                    skipped['decode_error'] += 1
                    continue
                
                # Get language
                ext = Path(rel_path).suffix.lower()
                language = self.CODE_EXTENSIONS.get(ext, 'text')
                
                # Get priority (for sorting only, not filtering!)
                priority = self.get_priority(rel_path)
                
                files.append({
                    'path': rel_path,
                    'code': content,
                    'language': language,
                    'priority': priority,
                    'size': info.file_size,
                    'lines': len(content.splitlines())
                })
        
        # Sort by priority (critical first, then high, medium, low)
        priority_order = {'critical': 0, 'high': 1, 'medium': 2, 'low': 3}
        files.sort(key=lambda x: (priority_order.get(x['priority'], 99), x['path']))
        
        return files, skipped
    
    def fetch_and_filter(self, github_url: str, max_files: Optional[int] = None) -> Dict:
        """
        Main method: fetch repo and return ALL valid files
        
        Args:
            github_url: GitHub repository URL
            max_files: Optional limit on number of files (None = no limit)
        """
        try:
            # Parse URL
            repo_info = self.parse_github_url(github_url)
            
            # Fetch ZIP
            zip_content = self.fetch_repo_zip(
                repo_info['owner'],
                repo_info['repo'],
                repo_info['branch']
            )
            
            # Extract ALL files
            all_files, skipped = self.extract_files(zip_content)
            
            # Apply max_files limit if specified
            if max_files:
                selected_files = all_files[:max_files]
                limited = True
            else:
                selected_files = all_files
                limited = False
            
            # Generate summary
            summary = {
                'total_found': len(all_files),
                'selected': len(selected_files),
                'limited': limited,
                'skipped': skipped,
                'by_priority': {},
                'by_language': {},
                'total_lines': sum(f['lines'] for f in selected_files)
            }
            
            for f in selected_files:
                priority = f['priority']
                lang = f['language']
                summary['by_priority'][priority] = summary['by_priority'].get(priority, 0) + 1
                summary['by_language'][lang] = summary['by_language'].get(lang, 0) + 1
            
            return {
                'success': True,
                'repo': f"{repo_info['owner']}/{repo_info['repo']}",
                'branch': repo_info['branch'],
                'files': selected_files,
                'summary': summary
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e)
            }

# ============================================
# CONFIGURATION
# ============================================
class AzureConfig:
    def __init__(self):
        self.api_key = os.getenv("AZURE_OPENAI_API_KEY", "")
        self.endpoint = os.getenv("AZURE_OPENAI_ENDPOINT", "")
        self.deployment = os.getenv("AZURE_OPENAI_DEPLOYMENT", "gpt-4")
        self.api_version = "2024-02-15-preview"
    
    def validate(self) -> bool:
        return bool(self.api_key and self.endpoint and self.deployment)

# ============================================
# TOOL SCHEMAS
# ============================================
TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "parse_flexible_requirements",
            "description": "Parse requirements from ANY format (bullet points, numbered, paragraphs, JSON)",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"}
                },
                "required": ["text"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "analyze_code_smart",
            "description": "Smart analysis: auto-detect code type and run appropriate checks. DSA→skip security, WebApp→all checks",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "file_path": {"type": "string"},
                    "language": {"type": "string"}
                },
                "required": ["code", "file_path", "language"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_requirements",
            "description": "Check if code implements requirements",
            "parameters": {
                "type": "object",
                "properties": {
                    "code_files": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "path": {"type": "string"},
                                "code": {"type": "string"},
                                "language": {"type": "string"}
                            },
                            "required": ["path", "code", "language"]
                        }
                    },
                    "requirements": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "text": {"type": "string"},
                                "description": {"type": "string"},
                                "type": {"type": "string"}
                            }
                        }
                    }
                },
                "required": ["code_files", "requirements"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_final_report",
            "description": "Generate PASS/FAIL report with reasons",
            "parameters": {
                "type": "object",
                "properties": {
                    "code": {"type": "string"},
                    "file_path": {"type": "string"},
                    "language": {"type": "string"},
                    "quality_issues": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "file": {"type": "string"},
                                "line": {"type": "integer"},
                                "message": {"type": "string"},
                                "severity": {"type": "string"},
                                "rule": {"type": "string"}
                            }
                        }
                    },
                    "security_issues": {
                        "type": "array",
                        "items": {"type": "object"}
                    },
                    "performance_issues": {
                        "type": "array",
                        "items": {"type": "object"}
                    },
                    "compliance_results": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "requirement_id": {"type": "string"},
                                "requirement": {"type": "string"},
                                "status": {"type": "string"},
                                "details": {"type": "string"}
                            }
                        }
                    }
                },
                "required": ["code", "file_path", "language", "quality_issues", "security_issues", "performance_issues", "compliance_results"]
            }
        }
    }
]

# ============================================
# AGENT TOOLS
# ============================================
class AgentTools:
    def __init__(self, ai_mode: str = "smart"):
        self.toolbox = AgentToolbox()
        self.enhanced_parser = EnhancedRequirementParser()
        self.smart_analyzer = SmartAnalyzer()
        self.ai_mode = ai_mode
        
        if LLM_AVAILABLE:
            self.llm = LLMCodeAnalyzer()
        else:
            self.llm = None
    
    def parse_flexible_requirements(self, text: str):
        result = self.enhanced_parser.parse(text)
        return {
            "success": True,
            "requirements": result["requirements"],
            "format": result["format_detected"],
            "count": result["total_count"]
        }
    
    def analyze_code_smart(self, code: str, file_path: str, language: str):
        analysis = self.smart_analyzer.analyze_code_type(code, file_path, language)
        
        result = {
            "success": True,
            "file": file_path,
            "code_type": analysis["code_type"],
            "reasoning": analysis["reasoning"],
            "checks_run": analysis["checks_needed"]
        }
        
        if analysis["checks_needed"]["quality"]:
            quality = self.toolbox.quality.analyze(code, file_path)
            result["quality_issues"] = quality
            result["quality_count"] = len(quality)
        
        if analysis["checks_needed"]["security"]:
            security = self.toolbox.security.analyze(code, file_path, language)
            result["security_issues"] = security
            result["security_count"] = len(security)
            result["security_critical"] = sum(1 for i in security if i.get('severity') == 'Error')
        else:
            result["security_issues"] = []
            result["security_skipped"] = True
            result["security_reason"] = f"Not needed for {analysis['code_type']}"
        
        result["semantic_issues"] = []
        if self.ai_mode != "off" and self.llm and self.llm.ollama_available:
            if analysis["checks_needed"]["performance"]:
                llm_result = self.llm.analyze_code_semantics(code, file_path, language)
                result["semantic_issues"] = llm_result.get("semantic_issues", [])
                result["llm_used"] = True
        
        return result
    
    def check_requirements(self, code_files: list, requirements: list):
        results = self.toolbox.compliance.check_compliance(code_files, requirements)
        return {
            "success": True,
            "results": results,
            "count": len(results)
        }
    
    def generate_final_report(self, code: str, file_path: str, language: str,
                            quality_issues: list, security_issues: list,
                            performance_issues: list, compliance_results: list):
        
        analysis = self.smart_analyzer.analyze_code_type(code, file_path, language)
        
        report = self.smart_analyzer.generate_pass_fail_report(
            code_type=analysis["code_type"],
            checks_run=analysis["checks_needed"],
            quality_issues=quality_issues,
            security_issues=security_issues,
            performance_issues=performance_issues,
            compliance_results=compliance_results
        )
        
        return {
            "success": True,
            **report
        }

# ============================================
# AGENT
# ============================================
class CodeReviewAgent:
    def __init__(self, config: AzureConfig, ai_mode: str = "smart"):
        self.config = config
        self.client = AzureOpenAI(
            api_key=config.api_key,
            api_version=config.api_version,
            azure_endpoint=config.endpoint
        )
        self.tools = AgentTools(ai_mode)
        self.messages = []
        self.max_iterations = 15
    
    def execute_tool(self, name: str, args: dict):
        try:
            func = getattr(self.tools, name)
            return func(**args)
        except Exception as e:
            return {"success": False, "error": str(e)}
    
    def run(self, user_request: str, context: dict = None):
        system_prompt = """You are a smart code review agent. Tools:

1. parse_flexible_requirements - Parse ANY format requirements
2. analyze_code_smart - Auto-detect code type and run appropriate checks
3. check_requirements - Verify requirement compliance
4. generate_final_report - Create PASS/FAIL report with reasons

WORKFLOW:
1. Parse requirements (if provided)
2. For each file: analyze_code_smart
3. Check requirements compliance
4. Generate final report with reasons

The smart analyzer:
- Detects code type (DSA, web app, API, etc.)
- Skips irrelevant checks (e.g., no security for DSA)
- Explains WHY each check was run/skipped
- Returns PASS/FAIL with specific reasons

Always be clear and explain findings."""

        self.messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": self._format_request(user_request, context)}
        ]
        
        for iteration in range(self.max_iterations):
            response = self.client.chat.completions.create(
                model=self.config.deployment,
                messages=self.messages,
                tools=TOOL_SCHEMAS,
                temperature=1.0
            )
            
            message = response.choices[0].message
            
            if message.tool_calls:
                self.messages.append({
                    "role": "assistant",
                    "content": message.content or "",
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        } for tc in message.tool_calls
                    ]
                })
                
                for tc in message.tool_calls:
                    args = json.loads(tc.function.arguments)
                    result = self.execute_tool(tc.function.name, args)
                    
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tc.id,
                        "content": json.dumps(result)
                    })
            else:
                return {
                    "success": True,
                    "response": message.content,
                    "iterations": iteration + 1
                }
        
        return {"success": False, "error": "Max iterations reached"}
    
    def _format_request(self, request: str, context: dict):
        msg = f"Request: {request}\n\n"

        if context:
            if context.get("requirements"):
                msg += f"Requirements ({len(context['requirements'])} total):\n"
                for req in context["requirements"]:
                    req_text = req.get("text") or req.get("description") or str(req)
                    msg += f"- {req_text}\n"
                msg += "\n"

            if context.get("code_files"):
                msg += f"Code Files ({len(context['code_files'])} total):\n"
                for f in context["code_files"]:
                    msg += f"\n=== {f['path']} ({f['language']}) ===\n"
                    msg += f.get("code", "")
                    msg += "\n"

            if context.get("ai_mode"):
                msg += f"\nAI mode: {context['ai_mode']}\n"

        return msg

# ============================================
# STREAMLIT UI
# ============================================
st.set_page_config(page_title="Code Review Agent", page_icon="🤖", layout="wide")

def main():
    st.title("🤖 Smart Code Review Agent")
    st.caption("Upload code, paste GitHub URL, and requirements - get intelligent PASS/FAIL analysis")
    
    # Initialize session state
    if 'code_files' not in st.session_state:
        st.session_state.code_files = []
    if 'requirements' not in st.session_state:
        st.session_state.requirements = []
    if 'ai_mode' not in st.session_state:
        st.session_state.ai_mode = 'smart'
    
    config = AzureConfig()
    
    # Sidebar
    with st.sidebar:
        st.header("🤖 AI Settings")
        
        # Check Ollama
        ollama_status = False
        if LLM_AVAILABLE:
            try:
                llm = LLMCodeAnalyzer()
                ollama_status = llm.ollama_available
            except:
                pass
        
        if ollama_status:
            st.success("● Ollama Online")
        else:
            st.warning("○ Ollama Offline")
        
        ai_mode = st.radio(
            "AI Mode:",
            ["off", "smart", "always"],
            index=1 if ollama_status else 0,
            format_func=lambda x: {
                "off": "❌ Off",
                "smart": "🧠 Smart (Recommended)",
                "always": "🚀 Always"
            }[x]
        )
        
        st.session_state.ai_mode = ai_mode
        
        st.divider()
        st.write(f"**Azure:** {'✅' if config.validate() else '❌'}")
        st.write(f"**LLM:** {'✅' if LLM_AVAILABLE else '❌'}")
        
        st.divider()
        st.header("⚙️ GitHub Settings")
        github_token = st.text_input("GitHub Token (optional)", type="password", help="For private repos or higher rate limits")
    
    # Validate config
    if not config.validate():
        st.error("❌ Set Azure OpenAI env vars:\n```\nAZURE_OPENAI_API_KEY\nAZURE_OPENAI_ENDPOINT\nAZURE_OPENAI_DEPLOYMENT\n```")
        return
    
    if not OPENAI_AVAILABLE:
        st.error("❌ Install: `pip install openai requests`")
        return
    
    # Main UI - Three columns
    col1, col2, col3 = st.columns([2, 2, 1])
    
    # Code Source Selection
    with col1:
        st.subheader("📝 Code Source")
        
        code_source = st.radio(
            "Choose source:",
            ["📁 Upload Files", "🔗 GitHub URL"],
            horizontal=True
        )
        
        if code_source == "📁 Upload Files":
            code_files = st.file_uploader(
                "Upload code files",
                type=['py', 'js', 'ts', 'jsx', 'tsx', 'java', 'cs', 'go', 'cpp', 'c', 'h', 'rs', 'rb', 'php'],
                accept_multiple_files=True
            )
            
            if code_files:
                processed_files = []
                for f in code_files:
                    code = f.read().decode('utf-8')
                    ext = f.name.split('.')[-1]
                    
                    lang_map = {
                        'py': 'python', 'js': 'javascript', 'ts': 'typescript',
                        'jsx': 'javascript', 'tsx': 'typescript',
                        'java': 'java', 'cs': 'csharp', 'go': 'go', 
                        'cpp': 'cpp', 'c': 'c', 'h': 'c', 'rs': 'rust',
                        'rb': 'ruby', 'php': 'php'
                    }
                    
                    processed_files.append({
                        "path": f.name,
                        "code": code,
                        "language": lang_map.get(ext, ext),
                        "priority": "high",
                        "lines": len(code.splitlines())
                    })
                
                st.session_state.code_files = processed_files
                st.success(f"✅ {len(processed_files)} file(s) loaded")
        
        else:  # GitHub URL
            github_url = st.text_input(
                "GitHub Repository URL",
                placeholder="https://github.com/owner/repo",
                help="Supports full repo URLs or specific branches/folders"
            )
            
            # FIXED: Allow unlimited files option
            max_files_option = st.radio(
                "File limit:",
                ["50 files", "100 files", "200 files", "All files"],
                index=3,  # Default to "All files"
                horizontal=True
            )
            
            max_files_map = {
                "50 files": 50,
                "100 files": 100,
                "200 files": 200,
                "All files": None
            }
            max_files = max_files_map[max_files_option]
            
            if github_url and st.button("🔍 Fetch Repository", type="secondary"):
                with st.spinner("Fetching repository..."):
                    fetcher = GitHubRepoFetcher(github_token)
                    result = fetcher.fetch_and_filter(github_url, max_files)
                    
                    if result['success']:
                        st.session_state.code_files = result['files']
                        
                        # Show summary
                        summary = result['summary']
                        
                        if summary['limited']:
                            st.warning(f"⚠️ Limited to {summary['selected']}/{summary['total_found']} files from **{result['repo']}** (branch: {result['branch']})")
                        else:
                            st.success(f"✅ Fetched ALL {summary['selected']} files from **{result['repo']}** (branch: {result['branch']})")
                        
                        with st.expander("📊 Repository Summary"):
                            col_s1, col_s2, col_s3 = st.columns(3)
                            with col_s1:
                                st.metric("Total Lines", f"{summary['total_lines']:,}")
                            with col_s2:
                                st.metric("Languages", len(summary['by_language']))
                            with col_s3:
                                st.metric("Files Selected", summary['selected'])
                            
                            st.write("**Skipped:**")
                            skipped = summary['skipped']
                            st.write(f"- Ignored (dependencies/IDE): {skipped['ignored']}")
                            st.write(f"- Binary/media files: {skipped['binary']}")
                            st.write(f"- Too large (>500KB): {skipped['too_large']}")
                            st.write(f"- Decode errors: {skipped['decode_error']}")
                            
                            st.write("**By Priority:**")
                            for priority, count in summary['by_priority'].items():
                                st.write(f"- {priority.capitalize()}: {count} files")
                            
                            st.write("**By Language:**")
                            for lang, count in sorted(summary['by_language'].items(), key=lambda x: -x[1]):
                                st.write(f"- {lang}: {count} files")
                            
                            st.write("**Sample Files:**")
                            for f in result['files'][:15]:
                                st.write(f"- `{f['path']}` ({f['priority']}, {f['lines']} lines, {f['language']})")
                            if len(result['files']) > 15:
                                st.write(f"... and {len(result['files']) - 15} more")
                    else:
                        st.error(f"❌ Error: {result['error']}")
    
    # Requirements
    with col2:
        st.subheader("📋 Requirements")
        
        req_format = st.radio(
            "Format:",
            ["Text (Any)", "JSON"],
            horizontal=True
        )
        
        if req_format == "Text (Any)":
            req_text = st.text_area(
                "Enter requirements:",
                height=200,
                placeholder="""Examples:
- User login with password
- Hash passwords with bcrypt
1. Implement authentication
2. Use JWT tokens"""
            )
            
            if req_text:
                parser = EnhancedRequirementParser()
                result = parser.parse(req_text)
                
                if result.get("requirements"):
                    st.session_state.requirements = result["requirements"]
                    st.success(f"✅ {result['total_count']} requirement(s) - {result['format_detected']}")
                    
                    with st.expander("View parsed"):
                        for r in result["requirements"]:
                            st.write(f"**{r['id']}** ({r['type']}): {r['text'][:80]}...")
        else:
            req_file = st.file_uploader("Upload JSON", type=['json'])
            if req_file:
                content = req_file.read().decode('utf-8')
                parser = EnhancedRequirementParser()
                result = parser.parse(content)
                
                if result.get("requirements"):
                    st.session_state.requirements = result["requirements"]
                    st.success(f"✅ {result['total_count']} requirement(s)")
    
    # Quick Stats
    with col3:
        st.subheader("📊 Status")
        
        if st.session_state.code_files:
            total_files = len(st.session_state.code_files)
            total_lines = sum(f.get('lines', 0) for f in st.session_state.code_files)
            st.metric("Files", total_files)
            st.metric("Lines", f"{total_lines:,}")
            
            # Language breakdown
            languages = {}
            for f in st.session_state.code_files:
                lang = f.get('language', 'unknown')
                languages[lang] = languages.get(lang, 0) + 1
            
            st.write("**Languages:**")
            for lang, count in sorted(languages.items(), key=lambda x: -x[1])[:5]:
                st.write(f"- {lang}: {count}")
        else:
            st.info("No code loaded")
        
        st.divider()
        
        if st.session_state.requirements:
            st.metric("Requirements", len(st.session_state.requirements))
        else:
            st.info("No requirements")
    
    st.divider()
    
    # Analysis
    if st.session_state.code_files and st.session_state.requirements:
        
        if st.button("🚀 Run Smart Analysis", type="primary", use_container_width=True):
            with st.spinner("🤖 Running intelligent code review..."):
                
                agent = CodeReviewAgent(config, st.session_state.ai_mode)
                
                context = {
                    "code_files": st.session_state.code_files,
                    "requirements": st.session_state.requirements,
                    "ai_mode": st.session_state.ai_mode
                }
                
                result = agent.run(
                    """Perform comprehensive smart code review:

1. Parse requirements using parse_flexible_requirements
2. For each file, use analyze_code_smart:
   - Auto-detect code type (DSA, web app, API, etc.)
   - Run appropriate checks based on context
   - Explain reasoning for each check
3. Check requirement compliance with check_requirements
4. Generate final PASS/FAIL report with generate_final_report

Present findings clearly with:
- Overall PASS/FAIL verdict
- Specific reasons for each check result
- Priority issues highlighted
- Actionable recommendations""",
                    context
                )
                
                if result["success"]:
                    st.markdown("## 📊 Analysis Results")
                    st.markdown(result["response"])
                    
                    col_r1, col_r2, col_r3 = st.columns(3)
                    with col_r1:
                        st.metric("Analysis Steps", result['iterations'])
                    with col_r2:
                        st.metric("AI Mode", st.session_state.ai_mode.upper())
                    with col_r3:
                        st.metric("Files Analyzed", len(st.session_state.code_files))
                    
                    st.success("✅ Analysis complete!")
                else:
                    st.error(f"❌ {result.get('error')}")
    
    else:
        st.info("👆 Load code (upload or GitHub) AND requirements to start")
        
        with st.expander("📖 How It Works"):
            st.markdown("""
### 🎯 Smart Context-Aware Analysis

This agent intelligently adapts analysis based on code type:

**For Algorithm/DSA Code:**
- ✅ Code quality checks
- ❌ Security checks skipped (not relevant)
- ✅ Performance analysis

**For Web Applications:**
- ✅ Code quality checks
- ✅ Security vulnerability scanning
- ✅ Performance analysis
- ✅ API endpoint validation

**For Libraries/Frameworks:**
- ✅ Code quality checks
- ✅ API design review
- ✅ Documentation completeness

### 🔍 GitHub Integration (FIXED!)

The fetcher now correctly:
- ✅ Processes ALL code files in ALL folders
- ✅ Only skips true junk (node_modules, binaries, etc.)
- ✅ Supports unlimited file fetching
- ✅ Shows detailed skip statistics
- ✅ Sorts by priority (critical/high/medium/low)

**What's Skipped:**
- Dependencies (node_modules, venv, etc.)
- Binary files (images, PDFs, executables)
- Large files (>500KB)
- Lock files (package-lock.json, etc.)

**What's Included:**
- ALL source code files (.py, .js, .ts, .java, etc.)
- Config files (.yaml, .json, etc.)
- Documentation (.md files)
- Scripts (.sh, .bash, etc.)

### 📋 Requirement Formats

Accepts any format:
- Bullet points (`- Feature X`)
- Numbered lists (`1. Feature X`)
- Plain paragraphs
- JSON structured data
            """)
        
        with st.expander("💡 Example Requirements"):
            st.markdown("""
**Security Requirements:**
- All passwords must be hashed using bcrypt
- API endpoints must use JWT authentication
- Input validation on all user inputs

**Functional Requirements:**
1. User registration with email verification
2. Password reset functionality
3. Profile management dashboard

**Performance Requirements:**
- API response time < 200ms
- Database queries optimized with indexes
- Caching implemented for frequent queries
            """)

if __name__ == "__main__":
    main()