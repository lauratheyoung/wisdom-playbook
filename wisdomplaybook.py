import gspread as gs
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from google.oauth2.service_account import Credentials
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import numpy as np
import base64


# Set-up
creds_info = st.secrets
scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
client = gs.authorize(creds)

# Open sheets
spreadsheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1XEIVMPSS69BHDBGPw8fKkuqze5iqXSP7JFfwdGQhSHk/edit#gid=406100282")
sheet = spreadsheet.worksheet("Individual")
peersheet = spreadsheet.worksheet("Peer Review")

#Load into dataframe
data = pd.DataFrame(sheet.get_all_records())
peerdata = pd.DataFrame(peersheet.get_all_records())

#Streamlit UI
st.set_page_config(layout="wide")
query_params = st.query_params
uuid_param = query_params.get("uuid", [None])[0]
uuid_input = st.text_input("Enter your report code:", value=uuid_param or "")

#load css

def load_css(file_name: str):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# Load CSS
load_css("styles.css")

#Load in font
st.markdown("""
    <link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">
""", unsafe_allow_html=True)

#PD Data Processing Functions
def split_user_data(df_user_data):
    i_cols = [col for col in df_user_data.columns if col.startswith("I ")]
    df_user_qs = df_user_data[i_cols]               # subset with "I " columns
    df_user_meta = df_user_data.drop(columns=i_cols)  # subset with all other columns
    return df_user_meta, df_user_qs

def split_peer_data(df_peer_data):
    they_cols = [col for col in df_peer_data.columns if col.startswith("They ")]
    df_peer_qs = df_peer_data[they_cols]               # subset with "I " columns
    df_peer_meta = df_peer_data.drop(columns=they_cols)  # subset with all other columns
    return df_peer_meta, df_peer_qs

def prepare_user_data(user_data_source):
    #Replace empty strings with NaNs
    df_user_data = user_data_source.replace(r'^\s*$', np.nan, regex=True).dropna(how='all')

    #Split dfs into Questions and Metadata subsets
    df_user_meta, df_user_qs = split_user_data(df_user_data)

    #Add formatted full name col
    full_name_format_user = lambda row: str(row.iloc[1] + " " + row.iloc[6]).lower()
    df_user_meta['formatted_fullname'] =  df_user_meta.apply(full_name_format_user, axis=1)

    return pd.concat([df_user_meta, df_user_qs], axis=1)

def prepare_peer_data(peer_data_source):
    #Replace empty strings with NaNs
    df_peer_data = peer_data_source.replace(r'^\s*$', np.nan, regex=True).dropna(how='all')

    #Split dfs into Questions and Metadata subsets
    df_peer_meta, df_peer_qs = split_peer_data(df_peer_data)

    #Add formatted full name col
    full_name_format_peer = lambda row: row.iloc[1].lower()
    df_peer_meta['formatted_fullname'] =  df_peer_meta.apply(full_name_format_peer, axis=1)

    return pd.concat([df_peer_meta, df_peer_qs], axis=1)

def get_user_row_by_uuid(user_data, uuid):
    return user_data[user_data['UUID'] == uuid].iloc[0]

def get_peer_data_from_user_row(peer_data, user_row):
    user_formatted_name = str(user_row['formatted_fullname'])
    return peer_data[peer_data['formatted_fullname'] == user_formatted_name]

def avg_peer_scores(peer_data):
    df_peer_qs = split_peer_data(peer_data)[1]
    return list(df_peer_qs.mean())


def get_user_scores_from_row(user_row):
    df_user_row = split_user_data(pd.DataFrame(user_row).T)[1]
    return list(df_user_row.iloc[0].astype(float))

#Initialising constant variables

# List of traits in order
TRAIT_COLS = ["Purposeful", "Playful", "Adventurous", "Adaptable",
              "Curious", "Charitable", "Engaged", "Ethical"]

# Column ranges for each trait (0-indexed for pandas)
TRAIT_RANGES = {
    "Purposeful": range(2, 6),
    "Playful": range(6, 10),
    "Adventurous": range(10, 14),
    "Adaptable": range(14, 18),
    "Curious": range(18, 22),
    "Charitable": range(22, 26),
    "Engaged": range(26, 30),
    "Ethical": range(30, 34),
}

# Backend logic to determine user trait scores (1 trait = 4 Questions) order of traits: Purposeful, playful, adventurous, adaptable, curious, charitable, engaged, ethical

def compute_trait_scores(df):
    df_traits = df.copy()
    df_traits.columns = df_traits.columns.str.strip()
    
    # Convert all relevant columns to numeric at once
    numeric_cols = sum([list(r) for r in TRAIT_RANGES.values()], [])
    df_traits.iloc[:, numeric_cols] = df_traits.iloc[:, numeric_cols].apply(pd.to_numeric, errors='coerce')

    for trait, col_range in TRAIT_RANGES.items():
        cols = df_traits.columns[list(col_range)]  # get actual column names
        # Convert to numeric safely (turn invalid entries into NaN)
        df_traits[cols] = df_traits[cols].apply(pd.to_numeric, errors='coerce')
        # Compute mean ignoring NaN
        df_traits[trait] = df_traits[cols].mean(axis=1)

    id_cols = ["Timestamp", "What is your first name?", "UUID"]
    id_cols = [col for col in id_cols if col in df_traits.columns]
    return df_traits[id_cols + TRAIT_COLS]

# Compute aggregated scores for all users and get df_traits
df_traits = compute_trait_scores(data)

# Compute for peer data
df_peer_traits = compute_trait_scores(peerdata)

def determine_strength_growth(user_row, trait_cols):
    """
    Determine the top 3 strength traits and bottom 2 growth traits for a given user,
    ensuring trait-score mapping is preserved and the fixed priority order is applied.

    Parameters:
        user_row: pandas Series representing a user's trait scores
        trait_cols: list of trait column names from the dataset

    Returns:
        strengths (list): exactly 3 top trait names
        growth (list): exactly 2 bottom trait names
    """

    # Desired order of traits for display/priority
    desired_order = ['Purposeful', 'Adventurous', 'Curious', 'Engaged',
                     'Playful', 'Adaptable', 'Charitable', 'Ethical']

    # Map user scores to the desired order
    traits_ordered = user_row.reindex(desired_order)

    # --- Top 3 strengths ---
    # Sort descending by score, preserving order in desired_order for ties
    strengths = traits_ordered.sort_values(ascending=False, kind='mergesort').head(3).index.tolist()

    # --- Bottom 2 growth traits ---
    # Sort ascending by score, preserving order in desired_order for ties
    growth = traits_ordered.sort_values(ascending=True, kind='mergesort').head(2).index.tolist()

    return strengths, growth

# --- Helper functions ---

def compute_peer_strengths(df, trait_cols):
    strengths_list, growth_list = [], []
    for _, row in df.iterrows():
        s, g = determine_strength_growth(row, trait_cols)
        strengths_list.append(", ".join(s))
        growth_list.append(", ".join(g))
    df["Peer_Strengths"] = strengths_list
    df["Peer_Growth"] = growth_list
    return df

def get_user_peer_feedback(peer_rows, trait_cols):
    if peer_rows.empty:
        return None, None, None
    peer_mean_scores = peer_rows[trait_cols].astype(float).mean()
    strengths, growth = determine_strength_growth(peer_mean_scores, trait_cols)
    return strengths, growth, peer_mean_scores

def compute_consistency(user_row, peer_mean_scores, trait_cols, tolerance=0.5):
    consistent, inconsistent = [], []
    if peer_mean_scores is None:
        return 0, consistent, inconsistent
    for trait in trait_cols:
        if abs(float(user_row[trait]) - float(peer_mean_scores[trait])) <= tolerance:
            consistent.append(trait)
        else:
            inconsistent.append(trait)
    pct = round(len(consistent) / len(trait_cols) * 100, 1)
    return pct, consistent, inconsistent

def display_dynamic_message(
    user_name, strengths, growth, 
    peer_strengths=None, peer_growth=None, 
    consistency_pct=None, consistent_traits=None, inconsistent_traits=None
):
    def get_base64_image(image_path):
        with open(image_path,"rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
        
    img_base64 = get_base64_image("assets/wisdomplaybook-logo.png")

    # Main container
    with st.container():

        st.markdown(
            f'''
            <div class="full-logo">
                <img class= "logo" src="data:image/png;base64,{img_base64}" alt="logo"/>
                <div class= "ful-logo-text">
                    <h2> The </h2>
                    <h2> Wisdom </h2>
                    <h2> Playbook </h2>
                </div>
            </div>
            ''',
            unsafe_allow_html=True
        )

        # peer text conditional
        peer_text = ""

        def format_trait_list(traits):
                if not traits:
                    return ""
                elif len(traits) == 1:
                    return traits[0]
                elif len(traits) == 2:
                    return f"{traits[0]} and {traits[1]}"
                else:
                    return f"{', '.join(traits[:-1])}, and {traits[-1]}"

        if peer_strengths and peer_growth and consistency_pct is not None:

            # Format trait lists properly
            formatted_consistent_traits = format_trait_list(consistent_traits)
            formatted_inconsistent_traits = format_trait_list(inconsistent_traits)

            inconsistent_text = (
                f"<p>There was a low consistency in how you assessed yourself and how your friends perceived you "
                f"for {consistency_pct}% of wisdom statements (for {formatted_consistent_traits}). "
                f"You may want to reflect on the inconsistencies between your own assessment and those of your friends, "
                f"in particular, these {formatted_inconsistent_traits} wisdom statements.</p>"
            )

            consistent_text = (
                f"<p>There was consistency in how you assessed yourself and how your friends perceived you "
                f"for {consistency_pct}% of wisdom statements (for {formatted_consistent_traits}). "
                f"You may want to reflect on the inconsistencies between your own assessment and those of your friends, "
                f"in particular, these {formatted_inconsistent_traits} wisdom statements.</p>"
            )

            no_consistent_text = (
                f"<p>There was no consistency in how you assessed yourself and how your friends perceived you "
                f"for {consistency_pct}% of wisdom statements. "
                f"You may want to reflect on the inconsistencies between your own assessment and those of your friends.</p>"
            )

            
            if 0 < consistency_pct <= 25:
                peer_text = inconsistent_text
            elif consistency_pct == 0:
                peer_text = no_consistent_text
            else:
                peer_text = consistent_text

        
        formatted_strengths = format_trait_list(strengths)
        formatted_growth = format_trait_list(growth)


        # Render a single card
        st.markdown(
            f'''
            <div class="congrats-card">
                <h1 class="congrats-text">Congratulations, {user_name}!</h1>
                <p>
                    You've taken the first steps towards reflecting on your own wisdom. 
                    Your Wisdom Report shows your areas of strength are: 
                    <span class="strengths">{formatted_strengths}</span> traits. 
                    The areas for you to work on are: 
                    <span class="growth">{formatted_growth}</span> traits.
                </p>
                {peer_text}
            </div>
            ''',
            unsafe_allow_html=True
        )

def plot_trait_comparison(user_row, peer_mean_scores, trait_cols):
    """
    user_row: pandas Series with individual user's trait scores
    peer_mean_scores: pandas Series with aggregated peer scores (can be None)
    trait_cols: list of traits in the order to plot
    """

    # Determine if peer data is available
    has_peer_data = peer_mean_scores is not None and not peer_mean_scores.empty

    # Desired display order
    display_order = [
        'Purposeful', 'Adventurous', 'Curious', 'Engaged',
        'Playful', 'Adaptable', 'Charitable', 'Ethical'
    ]

    # Reorder data according to display_order
    trait_cols = [trait for trait in display_order if trait in trait_cols]

    # Raw fractions for each trait (1–6 scale → 0–1)
    self_scores_raw = [user_row[trait] / 6 for trait in trait_cols]
    peer_scores_raw = [peer_mean_scores[trait] / 6 for trait in trait_cols] if has_peer_data else None

    # Convert to percentages
    self_scores = [round(s * 100, 1) for s in self_scores_raw]
    peer_scores = [round(p * 100, 1) for p in peer_scores_raw] if has_peer_data else None

    

    # Compute delta (in %), only if peer data exists
    delta_scores = (
    [round(peer - self_, 1) for self_, peer in zip(self_scores, peer_scores)]
    if has_peer_data else None
    )

    # Build horizontal bar chart
    fig = go.Figure()

    # --- Self-assessment bars ---
    fig.add_trace(go.Bar(
        y=trait_cols,
        x=self_scores,
        name='Self Assessment',
        orientation='h',
        marker_color='#898DF7',
        text=[f"{s}%" for s in self_scores],
        textposition='outside',
        hoverinfo='skip'
    ))

    # --- Peer bars (only if data exists) ---
    if has_peer_data:
        fig.add_trace(go.Bar(
            y=trait_cols,
            x=peer_scores,
            name='Peer Average',
            orientation='h',
            marker_color='#070D2E',
            text=[f"{p}%" for p in peer_scores],
            textposition='outside',
            hoverinfo='skip'
        ))

        # --- Delta annotations ---
        for i, trait in enumerate(trait_cols):
            fig.add_annotation(
                x=100,
                y=trait,
                text=f"Δ {delta_scores[i]}%",
                showarrow=False,
                font=dict(color='black', size=15),
                xanchor='right',
                yanchor='middle'
            )

        # Delta title
        fig.add_annotation(
            x=100,
            y=len(trait_cols),
            text="Score<br>Differences",
            showarrow=False,
            font=dict(color='black', size=15, family='Inter, sans-serif'),
            xanchor='right',
            yanchor='bottom'
        )

    fig.update_layout(
        barmode='group',
        font=dict(family='Inter, sans-serif'),
        title=dict(
            text='Your Wisdom Traits Assessment',
            font=dict(family='Inter', size=20, color='black')
        ),
        xaxis=dict(title='Score (%)', range=[0, 100]),
        height=50 * len(trait_cols) + 100,
        margin=dict(l=150, r=50, t=90, b=90),
        legend=dict(
            orientation='h',
            x=1,
            y=-0.3,
            xanchor='right',
            yanchor='bottom'
        ),
        yaxis=dict(
            title='',
            tickfont=dict(color='black', size=16),
            automargin=True,
            side='left',
            ticklabelposition='outside left',
            categoryorder='array',
            categoryarray=list(reversed(trait_cols))
        ),
    )

    return fig

def trait_plots(uuid, user_row, TRAIT_COLS, TRAIT_RANGES, user_peer_data):
    # Define the custom order
    desired_order = ['Purposeful', 'Adventurous', 'Curious', 'Engaged', 
                     'Playful', 'Adaptable', 'Charitable', 'Ethical']

    all_question_cols = split_user_data(pd.DataFrame(user_row).T)[1].columns
    all_question_scores = get_user_scores_from_row(user_row)

    if user_peer_data is not None and not user_peer_data.empty:
        all_peer_scores = avg_peer_scores(user_peer_data)
        has_peer = True
    else:
        all_peer_scores = [0] * len(all_question_cols)
        has_peer = False

    # Build a dictionary mapping trait -> its scores
    trait_scores_dict = {}
    col_ind_lower = 0
    for trait in TRAIT_COLS:
        col_ind_upper = col_ind_lower + 4
        trait_scores_dict[trait] = {
            "question_cols": all_question_cols[col_ind_lower:col_ind_upper],
            "question_scores": all_question_scores[col_ind_lower:col_ind_upper],
            "peer_scores": all_peer_scores[col_ind_lower:col_ind_upper] if has_peer else None
        }
        col_ind_lower += 4

    # Placeholder trait descriptions
    trait_descriptions = {
        "Purposeful": "You set clear goals and follow through with intention.",
        "Adventurous": "You enjoy exploring new ideas, people, and experiences.",
        "Curious": "You seek to learn and understand the world around you.",
        "Engaged": "You participate actively and invest attention in tasks.",
        "Playful": "You approach life with creativity and joy.",
        "Adaptable": "You adjust easily to new circumstances.",
        "Charitable": "You show generosity and care towards others.",
        "Ethical": "You act with integrity and follow moral principles."
    }

    # Loop through traits
    for trait in desired_order:
        data = trait_scores_dict[trait]
        question_cols = data["question_cols"]
        question_scores = data["question_scores"]
        peer_scores = data["peer_scores"]

        # --- Create pie chart ---
        def percent_of_max(scores):
            if not scores:
                return 0
            return round(sum(scores) / (4 * 6) * 100, 1)

        if has_peer and peer_scores is not None:
            combined_scores = [(s + p) / 2 for s, p in zip(question_scores, peer_scores)]
            overall_score = percent_of_max(combined_scores)
        else:
            overall_score = percent_of_max(question_scores)

        pie_fig = go.Figure(go.Pie(
            labels=[f"{trait} Score", " "],
            values=[overall_score, 100 - overall_score],
            hole=0.4,
            marker_colors=['#549D8A', '#D9D9D9'],
            textinfo='none',
            hoverinfo='skip',
            sort=False,
            rotation=180
        ))
        pie_fig.add_annotation(
            x=0.5, y=0.5,
            text=f"{round(overall_score, 1)}%",
            showarrow=False,
            font=dict(family='Inter, sans-serif', size=22, color='black')
        )
        pie_fig.update_layout(
            title=dict(text=f"{trait}", font=dict(family='Inter, sans-serif', size=25, color='black')),
            legend=dict(orientation='h', yanchor='top', xanchor='left', x=0.05)
        )

        # --- Create horizontal bar chart ---
        bar_fig = go.Figure()
        question_scores_pct = [(s / 6) * 100 for s in question_scores]

        bar_fig.add_trace(go.Bar(
            x=question_scores_pct,
            y=question_cols,
            orientation='h',
            name='Self Assessment',
            marker_color='#898DF7',
            text=[f"{round(s)}%" for s in question_scores_pct],
            textposition='outside'
        ))
        if has_peer:
            peer_scores_pct = [(s / 6) * 100 for s in peer_scores]
            bar_fig.add_trace(go.Bar(
                x=peer_scores_pct,
                y=question_cols,
                orientation='h',
                name='Peer Average',
                marker_color='#070D2E',
                text=[f"{round(s)}%" for s in peer_scores_pct],
                textposition='outside'
            ))

        bar_fig.update_layout(
            title_text=f"",
            title_font=dict(family='Inter, sans-serif', size=16, color='black'),
            xaxis=dict(range=[0, 105],
                       tickvals=[0, 20, 40, 60, 80, 100],
                       ticktext=["0%", "20%", "40%", "60%", "80%", "100%"]),
            barmode='group',
            margin=dict(b=80, t=50),
            legend=dict(
                orientation='h',
                x=1,
                y=-0.3,
                xanchor='right',
                yanchor='bottom',
                font=dict(size=13)
            )
        )

        def wrap_labels(labels, width=40):
            return ["<br>".join(textwrap.wrap(label, width=width)) for label in labels]
        bar_fig.update_traces(y=wrap_labels(question_cols, width=40), hoverinfo='skip')

        st.markdown(f"<div class='trait-window' style='display:flex; flex-wrap:wrap; background-color:#F7F7F7; border-radius:1.3rem; padding:1rem; margin-bottom:1rem; box-shadow:0 4px 8px rgba(0,0,0,0.1);'>", unsafe_allow_html=True)
        col1, col2 = st.columns([1, 2])
        with col1:
            st.plotly_chart(pie_fig, use_container_width=True, config={'displayModeBar':False})
        with col2:
            st.plotly_chart(bar_fig, use_container_width=True, config={'displayModeBar':False})
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("""
            <style>
            /* Hide the expander arrow text even on hover */
            div[data-testid="stExpander"] span[data-testid="stIconMaterial"] {
                display: none !important;
            }
            </style>
        """, unsafe_allow_html=True)


        with st.expander(f"{trait} — Click to see definition", icon="🌸"):
            st.markdown(f"""
                <div style='
                    background-color:#F7F7F7;
                    border-radius:1.3rem;
                    padding:1rem;
                    margin-top:0.5rem;
                    margin-bottom:1rem;
                    box-shadow:0 4px 8px rgba(0,0,0,0.1);
                    font-family: Inter, sans-serif;
                '>
                    {trait_descriptions.get(trait, "No definition available")}
                </div>
            """, unsafe_allow_html=True)






def dynamic_closing():
    st.markdown(f'''
        <div class="conclusion-card">
            <p>Your job now is to:</p>
            <p>1. Use your strong traits in how you show up; continue to share your strengths with the world.</p>
            <p>2. Explore ways of improving the scores on the traits and statements you scored lowest. The Wisdom Playbook offers you ways to do just that, working at your own pace.</p>
            <p>3. Reflect on the wisdom traits and statements where there was inconsistency between your own assessment and that of others.</p>
        </div>

        ''',
        unsafe_allow_html=True
    )

# --- Main logic ---

if not uuid_input:
    st.info("Enter or pass your report code in the URL to view your report.")
    st.stop()

user_data = data[data["UUID"] == uuid_input]
if user_data.empty:
    st.error("No report found for this report code.")
    st.stop()

df_traits["Full Name"] = data["What is your first name?"].str.strip() + " " + data["What is your last name?"].str.strip()
df_peer_traits["Full Name"] = peerdata["Who are you peer reviewing? (First and Last Name)"].str.strip()

user_traits = df_traits[df_traits["UUID"] == uuid_input]
if user_traits.empty:
    st.error("No trait data found for this report code.")
    st.stop()

# Compute user's strengths/growth
user_row = user_traits.iloc[0]
strengths, growth = determine_strength_growth(user_row, TRAIT_COLS)

# User's name
user_name = user_data["What is your first name?"].iloc[0]

# Peer computations
df_peer_traits = compute_peer_strengths(df_peer_traits, TRAIT_COLS)
peer_rows = df_peer_traits[df_peer_traits["Full Name"] == user_row["Full Name"]]
peer_strengths, peer_growth, peer_mean_scores = get_user_peer_feedback(peer_rows, TRAIT_COLS)
consistency_pct, consistent_traits, inconsistent_traits = compute_consistency(user_row, peer_mean_scores, TRAIT_COLS)

# Display message
display_dynamic_message(
    user_name, strengths, growth, 
    peer_strengths, peer_growth, 
    consistency_pct, consistent_traits, inconsistent_traits
)

# Load the overview chart
fig = plot_trait_comparison(user_row, peer_mean_scores, TRAIT_COLS)
st.plotly_chart(fig, use_container_width=True, config={'displayModeBar': False})


df_user_data = prepare_user_data(data)
df_peer_data = prepare_peer_data(peerdata)

user_row = get_user_row_by_uuid(df_user_data, uuid_input)
user_peer_data = get_peer_data_from_user_row(df_peer_data, user_row)

trait_plots(uuid_input, user_row, TRAIT_COLS, TRAIT_RANGES, user_peer_data)

dynamic_closing()