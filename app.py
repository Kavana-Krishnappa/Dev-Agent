"""
COMPLETE SIMPLIFIED CODE REVIEW AGENT
Smart analysis, flexible requirements, PASS/FAIL reporting
"""

import os
import json
from pathlib import Path
import streamlit as st

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
        # Step 1: Detect code type
        analysis = self.smart_analyzer.analyze_code_type(code, file_path, language)
        
        result = {
            "success": True,
            "file": file_path,
            "code_type": analysis["code_type"],
            "reasoning": analysis["reasoning"],
            "checks_run": analysis["checks_needed"]
        }
        
        # Step 2: Run quality (always)
        if analysis["checks_needed"]["quality"]:
            quality = self.toolbox.quality.analyze(code, file_path)
            result["quality_issues"] = quality
            result["quality_count"] = len(quality)
        
        # Step 3: Run security (if needed)
        if analysis["checks_needed"]["security"]:
            security = self.toolbox.security.analyze(code, file_path, language)
            result["security_issues"] = security
            result["security_count"] = len(security)
            result["security_critical"] = sum(1 for i in security if i.get('severity') == 'Error')
        else:
            result["security_issues"] = []
            result["security_skipped"] = True
            result["security_reason"] = f"Not needed for {analysis['code_type']}"
        
        # Step 4: LLM (if enabled and needed)
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
                temperature=1
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
            # =========================
            # REQUIREMENTS
            # =========================
            if context.get("requirements"):
                msg += f"Requirements ({len(context['requirements'])} total):\n"
                for req in context["requirements"]:
                    req_text = req.get("text") or req.get("description") or str(req)
                    msg += f"- {req_text}\n"
                msg += "\n"

            # =========================
            # CODE FILES
            # =========================
            if context.get("code_files"):
                msg += f"Code Files ({len(context['code_files'])} total):\n"
                for f in context["code_files"]:
                    msg += f"\n=== {f['path']} ({f['language']}) ===\n"
                    msg += f.get("code", "")
                    msg += "\n"

            # =========================
            # AI MODE
            # =========================
            if context.get("ai_mode"):
                msg += f"\nAI mode: {context['ai_mode']}\n"

        return msg

# ============================================
# STREAMLIT UI
# ============================================
st.set_page_config(page_title="Code Review Agent", page_icon="🤖", layout="wide")

def main():
    st.title("🤖 Smart Code Review Agent")
    st.caption("Upload code and requirements - get intelligent PASS/FAIL analysis")
    
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
    
    # Validate config
    if not config.validate():
        st.error("❌ Set Azure OpenAI env vars:\n```\nAZURE_OPENAI_API_KEY\nAZURE_OPENAI_ENDPOINT\nAZURE_OPENAI_DEPLOYMENT\n```")
        return
    
    if not OPENAI_AVAILABLE:
        st.error("❌ Install: `pip install openai`")
        return
    
    # Main UI
    col1, col2 = st.columns(2)
    
    # Code Upload
    with col1:
        st.subheader("📝 Code Files")
        code_files = st.file_uploader(
            "Upload code",
            type=['py', 'js', 'ts', 'java', 'cs', 'go', 'cpp'],
            accept_multiple_files=True
        )
        
        if code_files:
            processed_files = []
            for f in code_files:
                code = f.read().decode('utf-8')
                ext = f.name.split('.')[-1]
                
                lang_map = {
                    'py': 'python', 'js': 'javascript', 'ts': 'typescript',
                    'java': 'java', 'cs': 'csharp', 'go': 'go', 'cpp': 'cpp'
                }
                
                processed_files.append({
                    "path": f.name,
                    "code": code,
                    "language": lang_map.get(ext, ext)
                })
            
            st.session_state.code_files = processed_files
            st.success(f"✅ {len(processed_files)} file(s) loaded")
    
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
    
    st.divider()
    
    # Analysis
    if "code_files" in st.session_state and "requirements" in st.session_state:
        
        if st.button("🚀 Run Analysis", type="primary", use_container_width=True):
            with st.spinner("🤖 Analyzing..."):
                
                agent = CodeReviewAgent(config, st.session_state.ai_mode)
                
                context = {
                    "code_files": st.session_state.code_files,
                    "requirements": st.session_state.requirements,
                    "ai_mode": st.session_state.ai_mode
                }
                
                # Debug: show context being sent to the agent
                st.subheader("🔎 Debug: Context sent to agent")
                try:
                    st.json({
                        "code_files": st.session_state.get("code_files"),
                        "requirements": st.session_state.get("requirements"),
                        "ai_mode": st.session_state.get("ai_mode")
                    })
                except Exception:
                    st.write("(unable to render debug context)")

                result = agent.run(
                    """Perform smart code review:
1. Parse requirements using parse_flexible_requirements
2. For each file, use analyze_code_smart (auto-detects what to check)
3. Check requirements with check_requirements
4. Generate final report with generate_final_report

Show PASS/FAIL for each check with specific reasons.""",
                    context
                )
                
                if result["success"]:
                    st.markdown("## 📊 Results")
                    st.markdown(result["response"])
                    
                    col_r1, col_r2 = st.columns(2)
                    with col_r1:
                        st.metric("Analysis Steps", result['iterations'])
                    with col_r2:
                        st.metric("AI Mode", st.session_state.ai_mode.upper())
                    
                    st.success("✅ Analysis complete!")
                else:
                    st.error(f"❌ {result.get('error')}")
    
    else:
        st.info("👆 Upload code files AND requirements to start")
        
        with st.expander("📖 Example Requirements"):
            st.markdown("""
**Bullet format:**
- User must login with password
- Passwords hashed with bcrypt
- Use HTTPS for all connections

**Numbered format:**
1. Implement user authentication
2. Hash passwords securely
3. Use JWT tokens

**JSON format:**
```json
{
  "id": "FR-01",
  "description": "User authentication with secure passwords",
  "type": "security"
}
```
            """)

if __name__ == "__main__":
    main()