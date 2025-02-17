import streamlit as st
import pandas as pd
import requests
import json
from conversation_analytics_toolkit import wa_assistant_skills, transformation, filtering2 as filtering, analysis, visualization, selection as vis_selection, transcript, keyword_analysis

# Set up your credentials and URLs
api_key = 'pudoZJoHCNn9ivItgn0zDAfAJIdumm109ZFiN2Y2o0b5'
instance_id = '13e7edfa-8ca0-4aa2-a8f5-fff9eb761ba6'
assistant_id = '07ae41ec-e775-49c9-818e-298e430e7dc3'
version = '2023-05-01'
auth = ('apikey', api_key)

# URLs
base_url = f'https://api.us-south.assistant.watson.cloud.ibm.com/instances/{instance_id}/v2'
assistant_url = f'{base_url}/assistants/{assistant_id}?version={version}'
logs_url = f'{base_url}/assistants/{assistant_id}/logs?version={version}'

@st.cache_data
def fetch_data():
    # Fetch assistant details
    assistant_response = requests.get(assistant_url, auth=auth)
    assistant = assistant_response.json()
    
    print("Assistant Response:")
    print(json.dumps(assistant, indent=2))

    if 'error' in assistant:
        st.error(f"Error fetching assistant details: {assistant['error']}")
        st.stop()

    # Fetch logs
    logs_response = requests.get(logs_url, auth=auth)
    logs = logs_response.json()
    
    print("Logs Response:")
    print(json.dumps(logs, indent=2))

    if 'error' in logs:
        st.error(f"Error fetching logs: {logs['error']}")
        st.stop()

    # Check if 'logs' is in the response
    if 'logs' not in logs:
        st.warning("No 'logs' key found in the logs data. The response structure might have changed or there might be no logs available.")
        st.write("API Response for logs:")
        st.json(logs)
        return assistant, pd.DataFrame()  # Return an empty DataFrame if no logs

    return assistant, pd.DataFrame.from_records(logs['logs'])

def prepare_data(assistant, df_logs):
    assistant_skills = wa_assistant_skills.WA_Assistant_Skills()
    assistant_skills.add_skill(assistant["assistant_id"], assistant)
    df_logs_canonical = transformation.to_canonical_WA_v2(df_logs, assistant_skills, include_nodes_visited_str_types=True, include_context=True)
    return df_logs_canonical

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

def analyze_abandonments(df_logs_canonical):
    st.header("Analyzing Abandoned Conversations")
    st.subheader("Conversation Transcripts")
    milestone_selection = {"path": ["Appointment scheduling start", "Schedule time", "Enter zip code", "Branch selection"]}
    dropped_off_conversations = vis_selection.to_dataframe(milestone_selection)["dropped_off"]
    dfc = transcript.to_transcript(dropped_off_conversations)
    config = {'debugger': True}
    st.graphviz_chart(visualization.draw_transcript(config, dfc))

def identify_keywords_abandonment(df_logs_canonical):
    st.header("Identify Keywords at Point of Abandonment")
    milestone_selection = {"path": ["Appointment scheduling start", "Schedule time", "Enter zip code", "Branch selection"]}
    last_utterances_abandoned = vis_selection.get_last_utterances_from_selection(milestone_selection, df_logs_canonical)
    data = keyword_analysis.get_frequent_words_bigrams(last_utterances_abandoned, 10, 15, ["would", "pm", "ok", "yes", "no", "thank", "thanks", "hi", "i", "you"])
    config = {'flattened': True, 'width': 800, 'height': 500}
    st.graphviz_chart(visualization.draw_wordpackchart(config, data))

def main():
    st.title("IBM Watson Assistant Dialog Flow Analysis")
    st.sidebar.header("Navigation")
    options = ["Data Overview", "Visualize User Journeys", "Analyze Abandonments", "Identify Keywords"]
    choice = st.sidebar.radio("Go to", options)
    
    try:
        assistant, df_logs = fetch_data()
        
        if df_logs.empty:
            st.warning("No log data available. Some features may not work.")
        else:
            df_logs_canonical = prepare_data(assistant, df_logs)
        
        if choice == "Data Overview":
            st.subheader("Data Overview")
            if df_logs.empty:
                st.write("No log data available.")
            else:
                st.write(df_logs.head())
        elif choice == "Visualize User Journeys":
            if df_logs.empty:
                st.write("No log data available for visualization.")
            else:
                visualize_user_journeys(df_logs_canonical)
        elif choice == "Analyze Abandonments":
            if df_logs.empty:
                st.write("No log data available for abandonment analysis.")
            else:
                analyze_abandonments(df_logs_canonical)
        elif choice == "Identify Keywords":
            if df_logs.empty:
                st.write("No log data available for keyword identification.")
            else:
                identify_keywords_abandonment(df_logs_canonical)
    except Exception as e:
        st.error(f"An error occurred: {str(e)}")
        st.error("Please check the console for more detailed error information.")

if __name__ == '__main__':
    main()
