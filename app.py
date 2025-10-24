import os
import io
import time
import json
import yaml
import pandas as pd
import streamlit as st
from datetime import datetime
from typing import Dict, List, Any

from app.youtube import search_videos, fetch_transcript, fetch_top_comments
from app.llm_clients import LLMClient
from app.summarizer import summarize_video_text
from app.google_io import (
    get_google_clients, create_sheet_and_fill, 
    upload_text_as_gdoc, copy_doc_from_template_and_replace, 
    convert_upload_to_gdoc, create_custom_sheet
)
from app.template_engine import extract_text_from_template, apply_placeholders
from app.viz import kpi_bar, popularity_chart, tags_wordcloud, render_theme
from app.utils import ensure_state, as_markdown_table, to_download_json, load_agents_config
from app.themes import THEMES, apply_theme

# Page config
st.set_page_config(
    page_title="AI Research Agent Studio",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize state
ensure_state()

# Load agents config
agents_config = load_agents_config()

# Apply theme
current_theme = st.session_state.get("theme", "Midnight Blue")
apply_theme(current_theme)

# Sidebar
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=80)
    st.title("🚀 Agent Studio")
    
    # Theme selector
    st.subheader("🎨 Theme")
    selected_theme = st.selectbox(
        "Choose Theme",
        options=list(THEMES.keys()),
        index=list(THEMES.keys()).index(current_theme),
        key="theme_selector"
    )
    if selected_theme != current_theme:
        st.session_state.theme = selected_theme
        st.rerun()
    
    st.divider()
    
    # Navigation
    st.subheader("📍 Navigation")
    page = st.radio(
        "Select Mode",
        ["YouTube Research", "Custom Sheet Creator", "Agent Orchestrator"],
        index=st.session_state.get("page_index", 0),
        key="nav_radio"
    )
    st.session_state.page_index = ["YouTube Research", "Custom Sheet Creator", "Agent Orchestrator"].index(page)
    
    st.divider()
    
    # API Status
    st.subheader("🔑 API Status")
    def status_badge(key):
        return "🟢" if os.getenv(key) else "🔴"
    
    st.caption(f"{status_badge('YOUTUBE_API_KEY')} YouTube API")
    st.caption(f"{status_badge('GOOGLE_API_KEY')} Gemini API")
    st.caption(f"{status_badge('OPENAI_API_KEY')} OpenAI API")
    st.caption(f"{status_badge('GOOGLE_SERVICE_ACCOUNT_JSON')} Google Auth")
    
    st.divider()
    
    # Quick stats
    if st.session_state.videos:
        st.subheader("📊 Session Stats")
        st.metric("Videos Processed", len(st.session_state.videos))
        st.metric("Sheets Created", st.session_state.get("sheets_count", 0))
        st.metric("Docs Generated", st.session_state.get("docs_count", 0))

# Main content area
if page == "YouTube Research":
    render_youtube_research_page(agents_config)
elif page == "Custom Sheet Creator":
    render_custom_sheet_page(agents_config)
else:
    render_agent_orchestrator_page(agents_config)


def render_youtube_research_page(config):
    """Original YouTube research functionality with enhanced UI"""
    
    st.title("🎬 YouTube Research Agent")
    st.markdown("### Search, analyze, and export top YouTube videos with AI-powered summaries")
    
    # Input section with enhanced styling
    with st.container():
        col1, col2 = st.columns([3, 1])
        with col1:
            query = st.text_input(
                "🔍 Search Keywords",
                placeholder="e.g., AI agents, autonomous systems, machine learning",
                help="Enter keywords to search YouTube"
            )
        with col2:
            max_videos = st.number_input(
                "📊 Max Videos",
                min_value=5,
                max_value=20,
                value=20,
                step=1
            )
    
    # Advanced options
    with st.expander("⚙️ Advanced Options", expanded=False):
        col1, col2, col3 = st.columns(3)
        with col1:
            provider = st.selectbox(
                "Model Provider",
                ["gemini", "openai"],
                help="Select LLM provider"
            )
        with col2:
            models = config["agents"]["youtube_summarizer"]["models"][provider]
            model = st.selectbox(
                "Model",
                list(models.values()),
                help="Select specific model"
            )
        with col3:
            fetch_comments = st.toggle(
                "Fetch Comments",
                value=True,
                help="Include top comments in analysis"
            )
        
        temperature = st.slider(
            "Temperature",
            0.0, 1.0,
            config["agents"]["youtube_summarizer"]["parameters"]["temperature"],
            0.1
        )
    
    # Action buttons
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        run_btn = st.button("🚀 Search & Analyze", type="primary", use_container_width=True)
    with col2:
        if st.session_state.videos:
            st.download_button(
                "📥 Download JSON",
                data=to_download_json(st.session_state.videos),
                file_name=f"youtube_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
    with col3:
        if st.session_state.videos:
            clear_btn = st.button("🗑️ Clear Results", use_container_width=True)
            if clear_btn:
                st.session_state.videos = []
                st.session_state.results_df = None
                st.rerun()
    
    # Processing pipeline
    if run_btn:
        if not query.strip():
            st.error("⚠️ Please enter search keywords")
            st.stop()
        
        process_youtube_search(query, max_videos, provider, model, temperature, fetch_comments)
    
    # Display results
    if st.session_state.results_df is not None:
        display_youtube_results()


def process_youtube_search(query, max_videos, provider, model, temperature, fetch_comments):
    """Process YouTube search with progress tracking"""
    
    with st.status("🔄 Processing Pipeline", expanded=True) as status:
        # Step 1: Search
        st.write("🔍 Searching YouTube...")
        try:
            vids = search_videos(query, max_results=50)
            if not vids:
                st.error("No videos found")
                status.update(label="❌ Search Failed", state="error")
                return
            st.success(f"✅ Found {len(vids)} candidates")
        except Exception as e:
            st.error(f"Search error: {str(e)}")
            status.update(label="❌ Search Failed", state="error")
            return
        
        vids = vids[:max_videos]
        
        # Step 2: Initialize LLM
        st.write("🤖 Initializing AI model...")
        try:
            llm = LLMClient(provider=provider, model=model)
            st.success(f"✅ Using {provider}/{model}")
        except Exception as e:
            st.error(f"LLM initialization error: {str(e)}")
            status.update(label="❌ LLM Init Failed", state="error")
            return
        
        # Step 3: Process videos
        st.write("📝 Analyzing videos...")
        progress = st.progress(0)
        status_text = st.empty()
        
        rows = []
        for i, v in enumerate(vids):
            try:
                vid = v["videoId"]
                title = v["title"]
                url = f"https://www.youtube.com/watch?v={vid}"
                
                status_text.write(f"Processing [{i+1}/{len(vids)}]: {title[:50]}...")
                
                # Fetch transcript
                transcript = fetch_transcript(vid)
                fallback_text = v.get("description", "")
                
                # Fetch comments
                comments_list = []
                if fetch_comments:
                    try:
                        comments_list = fetch_top_comments(vid, max_comments=3)
                        if comments_list:
                            fallback_text += "\n\nTop comments:\n" + "\n".join(comments_list)
                    except:
                        pass
                
                # Generate summary
                summary = summarize_video_text(llm, transcript, fallback_text)
                
                row = {
                    "title": title,
                    "description": v.get("description", ""),
                    "summary_200w": summary,
                    "reference_link": url,
                    "comments": " | ".join(comments_list[:3]) if comments_list else "",
                    "viewCount": v.get("viewCount", 0),
                    "likeCount": v.get("likeCount", 0),
                    "commentCount": v.get("commentCount", 0),
                    "channelTitle": v.get("channelTitle", ""),
                    "publishedAt": v.get("publishedAt", ""),
                    "thumbnail": v.get("thumbnail", ""),
                    "tags": v.get("tags", []),
                    "duration": v.get("duration", ""),
                }
                rows.append(row)
                
            except Exception as e:
                st.warning(f"⚠️ Error processing video {i+1}: {str(e)}")
                continue
            
            progress.progress((i + 1) / len(vids))
        
        st.success(f"✅ Processed {len(rows)} videos")
        
        # Save results
        st.session_state.videos = rows
        st.session_state.results_df = pd.DataFrame(rows)
        st.session_state.json_data = rows
        
        status.update(label="✅ Pipeline Complete", state="complete", expanded=False)


def display_youtube_results():
    """Display YouTube research results with visualizations"""
    
    df = st.session_state.results_df
    
    st.divider()
    st.subheader("📊 Research Dashboard")
    
    # KPI metrics
    kpi_bar(df)
    
    # Tabs for different views
    tabs = st.tabs(["📸 Gallery", "📈 Analytics", "📋 Data Table", "🔧 Export Options"])
    
    with tabs[0]:
        st.markdown("### Video Gallery")
        cols = st.columns(4)
        for i, row in df.iterrows():
            with cols[i % 4]:
                st.image(row["thumbnail"], use_column_width=True)
                st.markdown(f"**[{row['title'][:60]}...]({row['reference_link']})**")
                st.caption(f"👁️ {row['viewCount']:,} | 👍 {row['likeCount']:,} | 💬 {row['commentCount']:,}")
                st.caption(f"📺 {row['channelTitle']}")
                with st.expander("Summary"):
                    st.write(row['summary_200w'])
    
    with tabs[1]:
        st.markdown("### Analytics")
        col1, col2 = st.columns(2)
        with col1:
            popularity_chart(df)
        with col2:
            tags_wordcloud(df["tags"].tolist())
    
    with tabs[2]:
        st.markdown("### Data Table")
        st.dataframe(
            df[["title", "channelTitle", "viewCount", "likeCount", "commentCount", "publishedAt", "reference_link"]],
            use_container_width=True,
            height=400
        )
    
    with tabs[3]:
        render_export_options(df)


def render_export_options(df):
    """Render export options for YouTube results"""
    
    st.markdown("### Export Options")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("#### Google Sheets")
        google_mode = st.selectbox("Auth Mode", ["service_account", "oauth"])
        
        if st.button("📊 Create Google Sheet", use_container_width=True):
            with st.spinner("Creating Google Sheet..."):
                try:
                    gc, sheets, docs, drive = get_google_clients(mode=google_mode)
                    sheet_id = create_sheet_and_fill(
                        gc,
                        f"YouTube Research - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        st.session_state.videos
                    )
                    st.session_state.sheet_id = sheet_id
                    st.session_state.sheets_count = st.session_state.get("sheets_count", 0) + 1
                    st.success(f"✅ [Open Sheet](https://docs.google.com/spreadsheets/d/{sheet_id})")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    with col2:
        st.markdown("#### Google Docs")
        st.caption("Upload a template with {{placeholders}}")
        
        template_file = st.file_uploader(
            "Upload Template",
            type=["txt", "md", "docx", "odt"],
            help="Use {{title}}, {{summary_200w}}, {{reference_link}}, etc."
        )
        
        doc_mode = st.radio(
            "Generation Mode",
            ["One doc per video", "Single compiled doc"],
            horizontal=True
        )
        
        if st.button("📄 Generate Docs", use_container_width=True):
            if not template_file:
                st.error("⚠️ Please upload a template")
            else:
                generate_docs_from_template(template_file, doc_mode, google_mode)


def render_custom_sheet_page(config):
    """New feature: Custom Google Sheet Creator"""
    
    st.title("📊 Custom Sheet Creator")
    st.markdown("### Design your own Google Sheet structure and populate it with data")
    
    # Step 1: Define columns
    st.subheader("Step 1: Define Sheet Structure")
    
    with st.expander("➕ Add Columns", expanded=True):
        col1, col2 = st.columns([2, 1])
        with col1:
            new_col = st.text_input("Column Name", placeholder="e.g., Product Name")
        with col2:
            if st.button("Add Column", use_container_width=True):
                if new_col:
                    if "custom_columns" not in st.session_state:
                        st.session_state.custom_columns = []
                    st.session_state.custom_columns.append(new_col)
                    st.rerun()
        
        if "custom_columns" in st.session_state and st.session_state.custom_columns:
            st.write("**Current Columns:**")
            for i, col in enumerate(st.session_state.custom_columns):
                col1, col2 = st.columns([4, 1])
                with col1:
                    st.text(f"{i+1}. {col}")
                with col2:
                    if st.button("❌", key=f"del_{i}"):
                        st.session_state.custom_columns.pop(i)
                        st.rerun()
    
    # Step 2: Upload template
    st.subheader("Step 2: Upload Document Template")
    
    template_file = st.file_uploader(
        "Upload template with {{placeholders}}",
        type=["txt", "md", "docx", "odt"],
        help="Use column names as placeholders: {{Product Name}}, etc."
    )
    
    template_text = st.text_area(
        "Or paste template here",
        height=200,
        placeholder="Example:\n# {{Product Name}}\n\nDescription: {{Description}}\n\nPrice: {{Price}}"
    )
    
    # Step 3: Configure agents
    st.subheader("Step 3: Configure AI Agents")
    
    render_agent_configuration(config)
    
    # Step 4: Generate
    st.subheader("Step 4: Generate Sheet & Docs")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        num_rows = st.number_input("Number of rows", 1, 100, 5)
    with col2:
        google_mode = st.selectbox("Google Auth", ["service_account", "oauth"])
    with col3:
        st.write("")  # spacing
        st.write("")  # spacing
        generate_btn = st.button("🚀 Generate", type="primary", use_container_width=True)
    
    if generate_btn:
        if not st.session_state.get("custom_columns"):
            st.error("⚠️ Please define at least one column")
        else:
            execute_custom_sheet_workflow(num_rows, template_file, template_text, google_mode, config)


def render_agent_configuration(config):
    """Render agent configuration UI"""
    
    if "selected_agents" not in st.session_state:
        st.session_state.selected_agents = []
    
    available_agents = list(config["agents"].keys())
    
    st.write("**Select Agents to Use:**")
    
    for agent_name in available_agents:
        agent_config = config["agents"][agent_name]
        
        with st.expander(f"🤖 {agent_name}", expanded=False):
            col1, col2 = st.columns([1, 3])
            
            with col1:
                use_agent = st.checkbox(
                    "Enable",
                    value=agent_name in st.session_state.selected_agents,
                    key=f"use_{agent_name}"
                )
                if use_agent and agent_name not in st.session_state.selected_agents:
                    st.session_state.selected_agents.append(agent_name)
                elif not use_agent and agent_name in st.session_state.selected_agents:
                    st.session_state.selected_agents.remove(agent_name)
            
            with col2:
                st.caption(agent_config["description"])
            
            if use_agent:
                # Provider selection
                provider = st.selectbox(
                    "Provider",
                    ["gemini", "openai"],
                    key=f"{agent_name}_provider"
                )
                
                # Model selection
                models = agent_config["models"][provider]
                model = st.selectbox(
                    "Model",
                    list(models.values()),
                    key=f"{agent_name}_model"
                )
                
                # Parameters
                col1, col2 = st.columns(2)
                with col1:
                    temp = st.slider(
                        "Temperature",
                        0.0, 1.0,
                        agent_config["parameters"]["temperature"],
                        0.1,
                        key=f"{agent_name}_temp"
                    )
                with col2:
                    max_tokens = st.number_input(
                        "Max Tokens",
                        100, 4000,
                        agent_config["parameters"]["max_tokens"],
                        100,
                        key=f"{agent_name}_tokens"
                    )
                
                # Custom prompt
                custom_prompt = st.text_area(
                    "Custom System Prompt (optional)",
                    value=agent_config["system_prompt"],
                    height=100,
                    key=f"{agent_name}_prompt"
                )


def execute_custom_sheet_workflow(num_rows, template_file, template_text, google_mode, config):
    """Execute custom sheet creation workflow"""
    
    with st.status("🔄 Executing Workflow", expanded=True) as status:
        # Step 1: Generate data using agents
        st.write("🤖 Running AI agents...")
        
        data_rows = []
        progress = st.progress(0)
        
        for i in range(num_rows):
            row_data = {}
            
            for col_name in st.session_state.custom_columns:
                # Use selected agents to generate data
                if st.session_state.selected_agents:
                    agent_name = st.session_state.selected_agents[0]  # Use first agent
                    agent_config = config["agents"][agent_name]
                    
                    provider = st.session_state.get(f"{agent_name}_provider", "gemini")
                    model = st.session_state.get(f"{agent_name}_model", "gemini-2.5-flash")
                    prompt = st.session_state.get(f"{agent_name}_prompt", agent_config["system_prompt"])
                    
                    try:
                        llm = LLMClient(provider=provider, model=model)
                        value = llm.complete(
                            prompt,
                            f"Generate a value for column '{col_name}' for row {i+1}. Return only the value, no explanation."
                        )
                        row_data[col_name] = value.strip()
                    except Exception as e:
                        row_data[col_name] = f"Error: {str(e)}"
                else:
                    row_data[col_name] = f"Sample data {i+1}"
            
            data_rows.append(row_data)
            progress.progress((i + 1) / num_rows)
        
        st.success(f"✅ Generated {len(data_rows)} rows")
        
        # Step 2: Create Google Sheet
        st.write("📊 Creating Google Sheet...")
        
        try:
            gc, sheets, docs, drive = get_google_clients(mode=google_mode)
            sheet_id = create_custom_sheet(
                gc,
                f"Custom Sheet - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                data_rows,
                list(st.session_state.custom_columns)
            )
            st.session_state.custom_sheet_id = sheet_id
            st.session_state.sheets_count = st.session_state.get("sheets_count", 0) + 1
            st.success(f"✅ [Open Sheet](https://docs.google.com/spreadsheets/d/{sheet_id})")
        except Exception as e:
            st.error(f"❌ Sheet creation error: {str(e)}")
            status.update(label="❌ Failed", state="error")
            return
        
        # Step 3: Generate docs if template provided
        if template_file or template_text:
            st.write("📄 Generating documents...")
            
            try:
                template_content = ""
                if template_file:
                    data = template_file.read()
                    template_content, mime = extract_text_from_template(data, template_file.name)
                else:
                    template_content = template_text
                
                doc_ids = []
                for idx, row in enumerate(data_rows):
                    doc_content = apply_placeholders(template_content, row)
                    doc_id = upload_text_as_gdoc(
                        drive, docs,
                        f"Doc {idx+1} - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        doc_content
                    )
                    doc_ids.append(doc_id)
                
                st.session_state.docs_count = st.session_state.get("docs_count", 0) + len(doc_ids)
                st.success(f"✅ Generated {len(doc_ids)} documents")
                
                for doc_id in doc_ids:
                    st.markdown(f"- [Document](https://docs.google.com/document/d/{doc_id})")
                
            except Exception as e:
                st.error(f"❌ Doc generation error: {str(e)}")
        
        status.update(label="✅ Workflow Complete", state="complete", expanded=False)


def render_agent_orchestrator_page(config):
    """Agent orchestrator for advanced workflows"""
    
    st.title("🎯 Agent Orchestrator")
    st.markdown("### Build and execute multi-agent workflows")
    
    st.info("🚧 Advanced orchestration features coming soon! This will allow you to chain multiple agents, create conditional workflows, and build complex automation pipelines.")
    
    # Preview of orchestrator features
    with st.expander("🔮 Planned Features", expanded=True):
        st.markdown("""
        - **Workflow Builder**: Visual drag-and-drop agent workflow designer
        - **Conditional Logic**: Add if/else conditions between agent steps
        - **Data Transformations**: Built-in data processing between agents
        - **Parallel Execution**: Run multiple agents simultaneously
        - **Error Handling**: Automatic retry and fallback strategies
        - **Workflow Templates**: Pre-built templates for common tasks
        - **Real-time Monitoring**: Live progress tracking and logs
        - **Webhook Integration**: Trigger workflows from external sources
        """)


def generate_docs_from_template(template_file, doc_mode, google_mode):
    """Generate Google Docs from template"""
    
    with st.spinner("Generating documents..."):
        try:
            gc, sheets, docs, drive = get_google_clients(mode=google_mode)
            
            # Extract template
            data = template_file.read()
            template_text, mime = extract_text_from_template(data, template_file.name)
            
            rows = st.session_state.videos
            doc_ids = []
            
            if "One doc per video" in doc_mode:
                # Create individual docs
                for idx, row in enumerate(rows):
                    mapping = {
                        "title": row.get("title", ""),
                        "description": row.get("description", ""),
                        "summary_200w": row.get("summary_200w", ""),
                        "reference_link": row.get("reference_link", ""),
                        "comments": row.get("comments", ""),
                        "channelTitle": row.get("channelTitle", ""),
                        "viewCount": str(row.get("viewCount", "")),
                        "likeCount": str(row.get("likeCount", "")),
                        "commentCount": str(row.get("commentCount", "")),
                    }
                    
                    content = apply_placeholders(template_text, mapping)
                    doc_id = upload_text_as_gdoc(
                        drive, docs,
                        f"{idx+1:02d} - {row.get('title', '')[:60]}",
                        content
                    )
                    doc_ids.append(doc_id)
            else:
                # Create compiled doc
                compiled = []
                for row in rows:
                    mapping = {
                        "title": row.get("title", ""),
                        "description": row.get("description", ""),
                        "summary_200w": row.get("summary_200w", ""),
                        "reference_link": row.get("reference_link", ""),
                        "comments": row.get("comments", ""),
                    }
                    section = apply_placeholders(template_text, mapping)
                    compiled.append(section)
                
                content = "\n\n---\n\n".join(compiled)
                doc_id = upload_text_as_gdoc(
                    drive, docs,
                    f"Compiled Research - {datetime.now().strftime('%Y%m%d_%H%M%S')}",
                    content
                )
                doc_ids.append(doc_id)
            
            st.session_state.docs_count = st.session_state.get("docs_count", 0) + len(doc_ids)
            st.success(f"✅ Generated {len(doc_ids)} document(s)")
            
            for doc_id in doc_ids:
                st.markdown(f"- [Open Document](https://docs.google.com/document/d/{doc_id})")
        
        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
