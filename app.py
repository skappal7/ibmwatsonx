import streamlit as st
import pandas as pd
import requests
import json
from conversation_analytics_toolkit import wa_assistant_skills, transformation, filtering2 as filtering, analysis, visualization, selection as vis_selection, transcript, keyword_analysis

# Set up your credentials and URLs
api_key = 'pudoZJoHCNn9ivItgn0zDAfAJIdumm109ZFiN2Y2o0b5'
instance_id = '13e7edfa-8ca0-4aa2-a8f5-fff9eb761ba6'
workspace_id = 'c41206c0-11dc-4d2b-afe5-25feea74d7fc'
version = '2021-06-14'
auth = ('apikey', api_key)

# URLs
workspace_url = f'https://api.us-south.assistant.watson.cloud.ibm.com/instances/{instance_id}/v1/workspaces/{workspace_id}?version={version}'
logs_url = f'https://api.us-south.assistant.watson.cloud.ibm.com/instances/{instance_id}/v1/workspaces/{workspace_id}/logs?version={version}'

# Function to fetch data
@st.cache_data
def fetch_data():
    # Fetch workspace
    workspace_response = requests.get(workspace_url, auth=auth)
    workspace = workspace_response.json()

    # Fetch logs
    logs_response = requests.get(logs_url, auth=auth)
    logs = logs_response.json()

    return workspace, pd.DataFrame.from_records(logs['logs'])

# Prepare data
def prepare_data(workspace, df_logs):
    assistant_skills = wa_assistant_skills.WA_Assistant_Skills()
    assistant_skills.add_skill(workspace["workspace_id"], workspace)
    df_logs_canonical = transformation.to_canonical_WA_v2(df_logs, assistant_skills, include_nodes_visited_str_types=True, include_context=True)
    return df_logs_canonical

# Visualize user journeys and abandonments
def visualize_user_journeys(df_logs_canonical):
    st.header("Visualizing User Journeys and Abandonments")
    turn_based_path_flows = analysis.aggregate_flows(df_logs_canonical, mode="turn-based", on_column="turn_label", max_depth=400, trim_reroutes=False)
    config = {
        'commonRootPathName': "All Conversations",
        'height': 800,
        'nodeWidth': 250,
        'maxChildrenInNode': 6,
        'linkWidth': 400,
        'sortByAttribute': 'flowRatio',
        'title': "All Conversations",
        'mode': "turn-based"
    }
    jsondata = json.loads(turn_based_path_flows.to_json(orient='records'))
    st.graphviz_chart(visualization.draw_flowchart(config, jsondata))

# Analyze abandoned conversations
def analyze_abandonments(df_logs_canonical):
    st.header("Analyzing Abandoned Conversations")
    st.subheader("Conversation Transcripts")
    milestone_selection = {"path": ["Appointment scheduling start", "Schedule time", "Enter zip code", "Branch selection"]}
    dropped_off_conversations = vis_selection.to_dataframe(milestone_selection)["dropped_off"]
    dfc = transcript.to_transcript(dropped_off_conversations)
    config = {'debugger': True}
    st.graphviz_chart(visualization.draw_transcript(config, dfc))

# Identify keywords at point of abandonment
def identify_keywords_abandonment(df_logs_canonical):
    st.header("Identify Keywords at Point of Abandonment")
    milestone_selection = {"path": ["Appointment scheduling start", "Schedule time", "Enter zip code", "Branch selection"]}
    last_utterances_abandoned = vis_selection.get_last_utterances_from_selection(milestone_selection, df_logs_canonical)
    data = keyword_analysis.get_frequent_words_bigrams(last_utterances_abandoned, 10, 15, ["would", "pm", "ok", "yes", "no", "thank", "thanks", "hi", "i", "you"])
    config = {'flattened': True, 'width': 800, 'height': 500}
    st.graphviz_chart(visualization.draw_wordpackchart(config, data))

# Streamlit App Layout
def main():
    st.title("IBM Watson Assistant Dialog Flow Analysis")
    st.sidebar.header("Navigation")
    options = ["Data Overview", "Visualize User Journeys", "Analyze Abandonments", "Identify Keywords"]
    choice = st.sidebar.radio("Go to", options)
    
    workspace, df_logs = fetch_data()
    df_logs_canonical = prepare_data(workspace, df_logs)
    
    if choice == "Data Overview":
        st.subheader("Data Overview")
        st.write(df_logs.head())
    elif choice == "Visualize User Journeys":
        visualize_user_journeys(df_logs_canonical)
    elif choice == "Analyze Abandonments":
        analyze_abandonments(df_logs_canonical)
    elif choice == "Identify Keywords":
        identify_keywords_abandonment(df_logs_canonical)

if __name__ == '__main__':
    main()
