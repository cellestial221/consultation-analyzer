import streamlit as st
import anthropic
import PyPDF2
import re
from typing import List, Dict, Tuple

# Configuration for analysis topics
ANALYSIS_TOPICS = {
    "PACCAR": {
        "description": "2023 Supreme Court's PACCAR ruling",
        "search_terms": ["PACCAR"],
        "prompt": "Analyse this consultation response document and provide a summary of the respondent's views on the 2023 Supreme Court's PACCAR ruling. Focus on their position, concerns, and recommendations related to this ruling."
    },
    "ALF Code": {
        "description": "The ALF Code",
        "search_terms": ["ALF Code"],
        "prompt": "Analyse this consultation response document and provide a summary of the respondent's views on the ALF Code. Focus on their position, concerns, and recommendations related to the ALF Code."
    },
    "Funding Regulation": {
        "description": "Funding regulation",
        "search_terms": ["regulation.*funding", "funding.*regulation"],
        "prompt": "Analyse this consultation response document and provide a summary of the respondent's views on funding regulation. Focus on their position, concerns, and recommendations related to how litigation funding should be regulated."
    },
    "Transparency/Disclosure": {
        "description": "Transparency/disclosure of funding arrangements",
        "search_terms": ["transparency", "disclosure"],
        "prompt": "Analyse this consultation response document and provide a summary of the respondent's views on transparency and disclosure of funding arrangements. Focus on their position, concerns, and recommendations related to disclosure requirements."
    },
    "Caps on Returns": {
        "description": "Caps on returns",
        "search_terms": ["caps"],
        "prompt": "Analyse this consultation response document and provide a summary of the respondent's views on caps on returns for litigation funders. Focus on their position, concerns, and recommendations related to return limitations."
    },
    "Capital Adequacy": {
        "description": "Capital adequacy",
        "search_terms": ["capital adequacy"],
        "prompt": "Analyse this consultation response document and provide a summary of the respondent's views on capital adequacy requirements for litigation funders. Focus on their position, concerns, and recommendations related to capital requirements."
    },
    "Recoverability of Damages": {
        "description": "Recoverability of damages",
        "search_terms": ["recoverable", "recoverability"],
        "prompt": "Analyse this consultation response document and provide a summary of the respondent's views on the recoverability of damages in litigation funding. Focus on their position, concerns, and recommendations related to damage recovery."
    },
    "Role of the Court": {
        "description": "Role of the court",
        "search_terms": ["role of the court"],
        "prompt": "Analyse this consultation response document and provide a summary of the respondent's views on the role of the court in litigation funding matters. Focus on their position, concerns, and recommendations related to judicial oversight and involvement."
    }
}

class ConsultationAnalyser:
    def __init__(self, api_key: str):
        self.client = anthropic.Anthropic(api_key=api_key)

    def extract_text_from_pdf(self, pdf_file) -> Tuple[str, Dict[int, str]]:
        """Extract text content from uploaded PDF file with page tracking."""
        try:
            pdf_reader = PyPDF2.PdfReader(pdf_file)
            full_text = ""
            page_texts = {}

            for page_num, page in enumerate(pdf_reader.pages, 1):
                page_text = page.extract_text()
                page_texts[page_num] = page_text
                full_text += f"\n--- PAGE {page_num} ---\n" + page_text + "\n"

            return full_text, page_texts
        except Exception as e:
            st.error(f"Error extracting text from PDF: {str(e)}")
            return "", {}

    def find_page_number(self, position: int, text: str) -> int:
        """Find which page a text position belongs to."""
        text_before = text[:position]
        page_markers = list(re.finditer(r'--- PAGE (\d+) ---', text_before))
        if page_markers:
            return int(page_markers[-1].group(1))
        return 1

    def search_text_excerpts(self, text: str, page_texts: Dict[int, str], search_terms: List[str], context_chars: int = 300) -> List[Dict]:
        """Search for terms in text and return excerpts with context and page numbers."""
        excerpts = []
        text_lower = text.lower()

        for term in search_terms:
            pattern = re.compile(term.lower(), re.IGNORECASE)
            matches = pattern.finditer(text_lower)

            for match in matches:
                page_num = self.find_page_number(match.start(), text)

                start = max(0, match.start() - context_chars)
                end = min(len(text), match.end() + context_chars)
                excerpt = text[start:end].strip()

                # Clean up page markers from excerpt
                excerpt = re.sub(r'\n--- PAGE \d+ ---\n', ' ', excerpt)

                # Highlight the matched term
                highlighted_excerpt = re.sub(
                    re.escape(term),
                    f"**{term}**",
                    excerpt,
                    flags=re.IGNORECASE
                )

                excerpt_data = {
                    'text': highlighted_excerpt,
                    'page': page_num,
                    'term': term
                }

                if not any(e['text'] == highlighted_excerpt for e in excerpts):
                    excerpts.append(excerpt_data)

        return excerpts

    def get_claude_analysis(self, text: str, prompt: str) -> Dict[str, str]:
        """Get analysis from Claude API."""
        try:
            # Limit text size for API
            max_chars = 90000
            if len(text) > max_chars:
                text = text[:max_chars] + "\n\n[Document truncated due to length...]"

            # Get detailed analysis
            detailed_message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                temperature=0.1,
                system="You are an expert legal analyst specialising in litigation funding regulation. Provide clear, concise analysis of consultation responses.",
                messages=[
                    {
                        "role": "user",
                        "content": f"{prompt}\n\nDocument content:\n{text}"
                    }
                ]
            )

            # Get formal summary
            topic_name = prompt.split('views on')[1].split('.')[0] if 'views on' in prompt else 'this topic'
            summary_prompt = f"""Based on the consultation response document, provide a formal 3-sentence summary about the respondent's position on {topic_name}.

            Format as: "The [respondent name] outlines that [key position]. [Main concern/recommendation]. [Conclusion/overall stance]."

            Make this suitable for copying into a professional email or report. Quote directly from the document where possible.

            Document content:\n{text}"""

            summary_message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=300,
                temperature=0.1,
                system="You are an expert legal analyst. Provide formal, professional summaries suitable for business communications.",
                messages=[
                    {
                        "role": "user",
                        "content": summary_prompt
                    }
                ]
            )

            return {
                "detailed_analysis": detailed_message.content[0].text,
                "formal_summary": summary_message.content[0].text
            }

        except Exception as e:
            return {
                "detailed_analysis": f"Error getting Claude analysis: {str(e)}",
                "formal_summary": f"Error getting formal summary: {str(e)}"
            }

    def analyse_topic(self, text: str, page_texts: Dict[int, str], topic_config: Dict) -> Dict:
        """Analyse a single topic in the document."""
        results = {
            "detailed_analysis": "",
            "formal_summary": "",
            "excerpts": [],
            "found_terms": []
        }

        # Get Claude analysis
        analysis_results = self.get_claude_analysis(text, topic_config["prompt"])
        results["detailed_analysis"] = analysis_results["detailed_analysis"]
        results["formal_summary"] = analysis_results["formal_summary"]

        # Search for excerpts
        excerpts = self.search_text_excerpts(text, page_texts, topic_config["search_terms"])
        results["excerpts"] = excerpts

        # Track which terms were found
        for term in topic_config["search_terms"]:
            if re.search(term.lower(), text.lower()):
                results["found_terms"].append(term)

        return results

def main():
    st.set_page_config(
        page_title="Consultation Response Analyser",
        page_icon="📄",
        layout="wide"
    )

    st.title("📄 Consultation Response Analyser")
    st.markdown("Analyse PDF consultation responses for key litigation funding topics using AI and text search.")

    # Initialize session state
    if "analysis_results" not in st.session_state:
        st.session_state.analysis_results = {}
    if "pdf_processed" not in st.session_state:
        st.session_state.pdf_processed = False
    if "current_file" not in st.session_state:
        st.session_state.current_file = None
    if "api_key" not in st.session_state:
        st.session_state.api_key = ""

    # API Key input
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input(
            "Enter your Claude API Key:",
            type="password",
            value=st.session_state.api_key,
            help="Get your API key from https://console.anthropic.com/settings/keys"
        )

        if api_key:
            st.session_state.api_key = api_key
            st.success("✅ API key configured")
        else:
            st.warning("⚠️ Please enter your Claude API key to proceed")

    # File upload
    uploaded_file = st.file_uploader(
        "Upload PDF Consultation Response",
        type=['pdf'],
        help="Upload the PDF consultation response document to analyse"
    )

    if not api_key:
        st.error("Please enter your Claude API key in the sidebar to continue.")
        return

    if uploaded_file is not None:
        # Check if this is a new file
        file_id = f"{uploaded_file.name}_{uploaded_file.size}"
        if st.session_state.current_file != file_id:
            # New file uploaded, reset cache
            st.session_state.analysis_results = {}
            st.session_state.pdf_processed = False
            st.session_state.current_file = file_id

        try:
            analyser = ConsultationAnalyser(api_key)

            # Extract text from PDF (only if not already processed)
            if not st.session_state.pdf_processed:
                with st.spinner("Extracting text from PDF..."):
                    pdf_text, page_texts = analyser.extract_text_from_pdf(uploaded_file)

                if not pdf_text.strip():
                    st.error("No text could be extracted from the PDF. Please ensure the PDF contains readable text.")
                    return

                # Store extracted text in session state
                st.session_state.pdf_text = pdf_text
                st.session_state.page_texts = page_texts
                st.session_state.pdf_processed = True

                st.success(f"✅ Successfully extracted {len(pdf_text):,} characters from PDF")

                # Warn about large documents
                if len(pdf_text) > 90000:
                    st.warning(f"📄 **Large Document Detected** ({len(pdf_text):,} characters)")
                    st.info("Document will be truncated for AI analysis but full text search will work normally.")
            else:
                pdf_text = st.session_state.pdf_text
                page_texts = st.session_state.page_texts
                st.success(f"✅ Using cached PDF data ({len(pdf_text):,} characters)")

            # Show document preview
            with st.expander("📖 Document Preview (First 1000 characters)"):
                preview_text = pdf_text.replace('\n--- PAGE ', '\n**PAGE ').replace(' ---\n', '**\n')
                st.text(preview_text[:1000] + "..." if len(preview_text) > 1000 else preview_text)

            # Analysis section
            st.header("🔍 Analysis Results")

            # Run analysis for all topics if not already cached
            if not st.session_state.analysis_results:
                st.info("🔄 Analysing all topics...")
                progress_bar = st.progress(0)
                total_topics = len(ANALYSIS_TOPICS)

                for i, (topic_name, topic_config) in enumerate(ANALYSIS_TOPICS.items()):
                    st.write(f"📋 Analysing: {topic_config['description']}")

                    with st.spinner(f"Getting AI analysis for {topic_config['description']}..."):
                        st.session_state.analysis_results[topic_name] = analyser.analyse_topic(
                            pdf_text, page_texts, topic_config
                        )

                    # Update progress
                    progress = (i + 1) / total_topics
                    progress_bar.progress(progress)

                progress_bar.empty()
                st.success("✅ All topics analysed successfully!")

            # Create tabs for each topic
            tab_names = list(ANALYSIS_TOPICS.keys())
            tabs = st.tabs(tab_names)

            for i, (topic_name, topic_config) in enumerate(ANALYSIS_TOPICS.items()):
                with tabs[i]:
                    st.subheader(f"Analysis: {topic_config['description']}")

                    # Get cached results
                    results = st.session_state.analysis_results.get(topic_name, {})

                    if not results:
                        st.error(f"No analysis results found for {topic_name}. Please try re-analysing the document.")
                        continue

                    # Overall Assessment (Formal Summary)
                    st.markdown("### 📋 Overall Assessment")
                    if results.get("formal_summary") and "Error" not in results["formal_summary"]:
                        st.info(results["formal_summary"])
                        if st.button(f"📋 Copy Assessment", key=f"copy_{topic_name}"):
                            st.code(results["formal_summary"], language=None)
                            st.success("Assessment ready to copy!")
                    else:
                        st.warning("No formal assessment available for this topic.")

                    # Display results in columns
                    col1, col2 = st.columns([2, 1])

                    with col1:
                        st.markdown("### 🤖 Detailed Analysis")
                        if results.get("detailed_analysis") and "Error" not in results["detailed_analysis"]:
                            st.markdown(results["detailed_analysis"])
                        else:
                            st.info("No detailed analysis available for this topic.")

                    with col2:
                        st.markdown("### 🔍 Search Results")

                        if results.get("found_terms"):
                            st.success(f"Found {len(results['found_terms'])} relevant term(s):")
                            for term in results["found_terms"]:
                                st.markdown(f"- `{term}`")
                        else:
                            st.warning("No search terms found in document.")

                        if results.get("excerpts"):
                            st.markdown(f"**{len(results['excerpts'])} excerpt(s) found:**")
                        else:
                            st.info("No excerpts found for this topic.")

                    # Display excerpts with page numbers
                    if results.get("excerpts"):
                        st.markdown("### 📝 Document Excerpts")
                        for j, excerpt_data in enumerate(results["excerpts"], 1):
                            with st.expander(f"Excerpt {j} (Page {excerpt_data['page']})"):
                                st.markdown(f"**Page {excerpt_data['page']}** - Found: `{excerpt_data['term']}`")
                                st.markdown(excerpt_data['text'])

                    st.divider()

            # Summary section
            st.header("📊 Summary")

            topics_with_ai_analysis = 0
            topics_with_excerpts = 0

            for topic_name in ANALYSIS_TOPICS.keys():
                if topic_name in st.session_state.analysis_results:
                    results = st.session_state.analysis_results[topic_name]
                    if results.get("detailed_analysis") and "Error" not in results["detailed_analysis"]:
                        topics_with_ai_analysis += 1
                    if results.get("excerpts"):
                        topics_with_excerpts += 1

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Topics Analysed", len(ANALYSIS_TOPICS))
            with col2:
                st.metric("Topics with AI Analysis", topics_with_ai_analysis)
            with col3:
                st.metric("Topics with Text Excerpts", topics_with_excerpts)

            # Clear cache button
            if st.button("🔄 Re-analyse Document", help="Clear cache and re-run analysis"):
                st.session_state.analysis_results = {}
                st.session_state.pdf_processed = False
                st.rerun()

        except Exception as e:
            st.error(f"An error occurred during analysis: {str(e)}")
            st.error("Please check that your API key is correct and try again.")

    else:
        st.info("👆 Please upload a PDF consultation response to begin analysis.")

        # Show topic overview
        st.header("📋 Analysis Topics")
        st.markdown("This tool will analyse the following topics:")

        for i, (topic_name, topic_config) in enumerate(ANALYSIS_TOPICS.items(), 1):
            st.markdown(f"**{i}. {topic_config['description']}**")
            st.markdown(f"   - Search terms: {', '.join(topic_config['search_terms'])}")

if __name__ == "__main__":
    main()
