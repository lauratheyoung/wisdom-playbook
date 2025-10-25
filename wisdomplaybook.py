import gspread as gs
import plotly.graph_objects as go
from google.oauth2.service_account import Credentials
import streamlit as st
import pandas as pd
import streamlit.components.v1 as components

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
        df_traits[trait] = df_traits[cols].mean(axis=1).round(1)

    id_cols = ["Timestamp", "What is your first name?", "UUID"]
    id_cols = [col for col in id_cols if col in df_traits.columns]
    return df_traits[id_cols + TRAIT_COLS]


# Compute aggregated scores for all users and get df_traits
df_traits = compute_trait_scores(data)
# Compute for peer data
df_peer_traits = compute_trait_scores(peerdata)


def compute_peer_question_scores(individual_df: pd.DataFrame,
                                 peer_df: pd.DataFrame,
                                 question_col_start: int = 2,
                                 question_col_end: int = 34,
                                 first_name_col: str = "What is your first name?",
                                 last_name_col: str = "What is your last name?",
                                 peer_name_col: str = "Who are you peer reviewing? (First and Last Name)") -> pd.DataFrame:
    """
    Compute average scores per question for peers, matched by full name.

    Parameters:
        individual_df: DataFrame with individual responses (first + last name split)
        peer_df: DataFrame with peer responses (full name in one column)
        question_col_start, question_col_end: numeric question column slice (0-indexed)
        first_name_col, last_name_col: columns in individual_df
        peer_name_col: column in peer_df containing full name

    Returns:
        DataFrame: index = Full Name, columns = numeric question columns, values = average scores
    """
    ind_df = individual_df.copy()
    peer_df = peer_df.copy()
    
    # Normalize individual full name
    ind_df["Full Name"] = (ind_df[first_name_col].str.strip() + " " + ind_df[last_name_col].str.strip()).str.upper()
    
    # Normalize peer full name
    peer_df["Full Name"] = peer_df[peer_name_col].str.strip().str.upper()
    
    # Select question columns
    question_cols = peer_df.columns[question_col_start:question_col_end]
    
    # Convert to numeric
    peer_df[question_cols] = peer_df[question_cols].apply(pd.to_numeric, errors='coerce')
    
    # Group by full name to compute average per question
    peer_question_scores = peer_df.groupby("Full Name")[question_cols].mean().round(1)
    
    return peer_question_scores

peer_question_scores=compute_peer_question_scores(data, peerdata)

st.write(peer_question_scores.head())

# Backend logic to determine users strength and growth traits by aggregating trait scores and comparing  --> not sure what to do if there are ties
def determine_strength_growth(user_row, trait_cols, top_n=3):
    """
    Determine the top and bottom traits for a given user.

    Parameters:
        user_row: pandas Series representing a user's trait scores
        trait_cols: list of trait column names
        top_n: number of traits to include for strengths and growth areas

    Returns:
        strengths (list): top N trait names
        growth (list): bottom N trait names
    """
    # Extract only the numeric trait values
    traits = user_row[trait_cols].astype(float)

    # Sort traits descending for strengths, ascending for growth
    sorted_traits = traits.sort_values(ascending=False)

    # Get top N strengths (including ties)
    strengths_cutoff = sorted_traits.iloc[top_n - 1]
    strengths = sorted_traits[sorted_traits >= strengths_cutoff].index.tolist()

    # Get bottom N growth traits (including ties)
    growth_cutoff = sorted_traits.iloc[-top_n]
    growth = sorted_traits[sorted_traits <= growth_cutoff].index.tolist()

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

def compute_consistency(user_row, peer_mean_scores, trait_cols, tolerance=1.0):
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
    # Main container
    with st.container():
        st.markdown(
            f'''
            <div class="welcome-card">
                <h2>
                    Welcome <span class="user-name">{user_name}</span> to the 
                    <span class="wisdom-playbook">Wisdom Playbook</span>
                </h2>
            </div>
            ''',
            unsafe_allow_html=True
        )

        # Congrats card
        st.markdown(
            f'''
            <div class="congrats-card">
                <h1>Congratulations, {user_name}!</h1>
                <p>Your strength traits are: 
                    <span class="strengths">{', '.join(strengths)}</span>.
                </p>
                <p>Your growth traits are: 
                    <span class="growth">{', '.join(growth)}</span>.
                </p>
            </div>
            ''',
            unsafe_allow_html=True
        )

        # Peer card (optional)
        if peer_strengths and peer_growth and consistency_pct is not None:
            st.markdown(
                f'''
                <div class="peer-card">
                    <p>Consistency with peer assessment: {consistency_pct}% ({', '.join(consistent_traits)})</p>
                    <p>Inconsistencies: {', '.join(inconsistent_traits)}</p>
                </div>
                ''',
                unsafe_allow_html=True
            )


def plot_trait_comparison(user_row, peer_mean_scores, trait_cols):
    """
    user_row: pandas Series with individual user's trait scores
    peer_mean_scores: pandas Series with aggregated peer scores
    trait_cols: list of traits in the order to plot

    """
    # Extract and normalize scores (convert 0–6 scale to %) and round to 1 decimal
    self_scores = [round((user_row[trait] / 6) * 100, 1) for trait in trait_cols]
    peer_scores = [round((peer_mean_scores[trait] / 6) * 100, 1) if peer_mean_scores is not None else 0 for trait in trait_cols]

    # Compute delta (in %)
    delta_scores = [round(peer - self_, 1) for self_, peer in zip(self_scores, peer_scores)]

    # Build horizontal bar chart
    fig = go.Figure()

    # Individual self-assessment bars
    fig.add_trace(go.Bar(
        y=trait_cols,
        x=self_scores,
        name='Self Assessment',
        orientation='h',
        marker_color= '#898DF7',
        text=[f"{s}%" for s in self_scores],
        textposition='outside'
    ))

    # Peer assessment bars
    fig.add_trace(go.Bar(
        y=trait_cols,
        x=peer_scores,
        name='Peer Review',
        orientation='h',
        marker_color='#070D2E',
        text=[f"{p}%" for p in peer_scores],
        textposition='outside'
    ))

    # Add delta annotation
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
    fig.add_annotation(
    x=100,  # align with the delta scores column
    y=len(trait_cols),  # just above the top-most bar
    text="Score<br>Differences",
    showarrow=False,
    font=dict(color='black', size=15, family='Inter, sans-serif'),
    xanchor='right',
    yanchor='bottom'
)

    fig.update_layout(
        barmode='group',
        font=dict(family='Inter, sans-serif'),
        title=dict(text='Self vs Peer Wisdom Traits Assessment',
                   font=dict(family='Inter, sans-serif',size=20,color='black')),
        xaxis=dict(title='Score (%)', range=[0, 100]),
        height=50*len(trait_cols) + 100,
        margin=dict(l=150, r=50, t=40, b=100),  # increase bottom margin for legend
        legend=dict(
            orientation='h',
            y=-0.2,  # position below the x-axis
            x=0,
            xanchor='left',
            yanchor='top'
        ),
        yaxis=dict(
        title='',                    
        tickfont=dict(color='black', size=16),  
        automargin=True,
        side='left',                   # keep on left side
        ticklabelposition='outside left'
        ),
    )

    return fig

def trait_plots(uuid, data, TRAIT_COLS, TRAIT_RANGES, peer_data=None):
    """
    Generate pie chart for overall trait score and horizontal bar chart per question
    for each trait, comparing individual vs peer scores.
    peer_data: pandas DataFrame with same columns as `data` containing peer averages
    """
    user_row = data[data["UUID"] == uuid]
    
    if user_row.empty:
        st.error("No data found for this UUID.")
        return
    
    user_row = user_row.iloc[0]  # convert to Series
    
    for trait in TRAIT_COLS:
        raw_range = TRAIT_RANGES.get(trait)
        if not raw_range:
            continue
        
        if all(isinstance(i, int) for i in raw_range):
            question_cols = [data.columns[i] for i in raw_range]
        else:
            question_cols = list(raw_range)
        
        question_scores = pd.to_numeric(user_row[question_cols], errors='coerce').fillna(0).tolist()

        # --- Get peer scores if provided ---
        st.write('show')
        st.write(peer_data)
        st.write(question_cols)
        st.write(peer_data.iloc[:, :4])
        if peer_data is not None:
            peer_scores = pd.to_numeric(peer_data[question_cols].mean(), errors='coerce').fillna(0).tolist()
        else:
            peer_scores = [0] * len(question_cols)
        
        # --- Create grouped horizontal bar chart ---
        bar_fig = go.Figure()
        
        # Individual scores
        bar_fig.add_trace(go.Bar(
            x=question_scores,
            y=question_cols,
            orientation='h',
            name='Self',
            marker_color='#898DF7',
            text=[str(round(s,1)) for s in question_scores],
            textposition='outside'
        ))
        
        # Peer scores
        bar_fig.add_trace(go.Bar(
            x=peer_scores,
            y=question_cols,
            orientation='h',
            name='Peer Avg',
            marker_color='#FFA500',
            text=[str(round(s,1)) for s in peer_scores],
            textposition='outside'
        ))
        
        bar_fig.update_layout(
            title_text=f"{trait} - Question Scores",
            title_font=dict(family='Inter, sans-serif', size=16, color='black'),
            xaxis=dict(title="Score", range=[0, 7]),
            yaxis=dict(title="", tickfont=dict(family='Inter, sans-serif', size=10, color='black'), automargin=True),
            barmode='group',  # group bars side by side
            font=dict(family='Inter, sans-serif')
        )
        
        # --- Create pie chart for self score ---
        overall_score = sum(question_scores) / len(question_scores) if len(question_scores) > 0 else 0
        pie_fig = go.Figure(go.Pie(
            labels=[f"{trait} Score", "Remaining"],
            values=[overall_score, 6 - overall_score],
            hole=0.4,
            marker_colors=['#549D8A', '#D9D9D9'],
            textinfo='none'
        ))
        pie_fig.add_annotation(
            x=0.5,
            y=0.5,
            text=f"{round((overall_score/6)*100,1)}%",
            showarrow=False,
            font=dict(family='Inter, sans-serif', size=24, color='black')
        )
        pie_fig.update_layout(
            title=dict(
                text=f"{trait}",
                font=dict(family='Inter, sans-serif', size=25, color='black')
            ),
            legend=dict(
                orientation='h',
                y=-0.2,
                x=0.5,
                xanchor='center',
                yanchor='top',
                font=dict(family='Inter, sans-serif', size=12, color='black')
            )
        )
        
        # --- Display charts side by side ---
        col1, col2 = st.columns([1, 2])
        with col1:
            st.plotly_chart(pie_fig, use_container_width=True, config={'displayModeBar':False})
        with col2:
            st.plotly_chart(bar_fig, use_container_width=True, config={'displayModeBar':False})



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
st.dataframe(user_traits[TRAIT_COLS].T.rename(columns={user_traits.index[0]: "Individual Score"}))


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
st.plotly_chart(fig, use_container_width=True)

trait_plots(uuid_input, data, TRAIT_COLS, TRAIT_RANGES,peer_question_scores)