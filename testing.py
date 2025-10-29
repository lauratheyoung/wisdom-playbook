# import streamlit as st
# import pandas as pd
# import numpy as np
# from typing import List, Tuple, Optional, Dict
# from google.oauth2.service_account import Credentials
# import gspread
# import plotly.graph_objects as go
# import streamlit.components.v1 as components

# # -------------------------
# # Config / Constants
# # -------------------------
# SPREADSHEET_URL = "https://docs.google.com/spreadsheets/d/1XEIVMPSS69BHDBGPw8fKkuqze5iqXSP7JFfwdGQhSHk"
# INDIV_SHEET_NAME = "Individual"
# PEER_SHEET_NAME = "Peer Review"
# TRAITS = ["Purposeful", "Playful", "Adventurous", "Adaptable",
#           "Curious", "Charitable", "Engaged", "Ethical"]

# # Map trait -> list of question column headers (prefer explicit names)
# # Replace these keys with the exact header strings from your sheet.
# TRAIT_QUESTION_RANGES = {
#     "Purposeful": (3, 6),     # Q3, Q4, Q5, Q6
#     "Playful": (7, 10),       # Q7, Q8, Q9, Q10
#     "Adventurous": (11, 14),  # Q11, Q12, Q13, Q14
#     "Adaptable": (15, 18),    # Q15, Q16, Q17, Q18
#     "Curious": (19, 21),      # Q19, Q20, Q21
#     "Charitable": (22, 25),   # Q22, Q23, Q24, Q25
#     "Engaged": (26, 29),      # Q26, Q27, Q28, Q29
#     "Ethical": (30, 33),      # Q30, Q31, Q32, Q33
# }

# # -------------------------
# # Google client & fetch (cached)
# # -------------------------
# @st.cache_resource
# def get_gspread_client(_creds_info: dict):
#     scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
#     creds = Credentials.from_service_account_info(_creds_info, scopes=scopes)
#     return gspread.authorize(creds)

# @st.cache_data(ttl=300)  # cache sheet content for 5min (adjust TTL as needed)
# def fetch_sheet_df(_client: gspread.client.Client, sheet_name: str) -> pd.DataFrame:
#     """
#     Fetch worksheet by name from the same spreadsheet URL and return DataFrame.
#     """
#     sh = _client.open_by_url(SPREADSHEET_URL)
#     try:
#         ws = sh.worksheet(sheet_name)
#     except Exception as e:
#         raise RuntimeError(f"Could not open worksheet '{sheet_name}': {e}")
#     df = pd.DataFrame(ws.get_all_records())
#     # normalize column names (strip)
#     df.columns = df.columns.str.strip()
#     return df

# # -------------------------
# # Data processing functions
# # -------------------------
# def ensure_numeric(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
#     """Convert columns to numeric once; return df copy with conversions."""
#     df = df.copy()
#     df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
#     return df

# def compute_trait_scores(df: pd.DataFrame, trait_cols: List[str]) -> pd.DataFrame:
#     df = df.copy()
#     # ensure numeric
#     df[trait_cols] = df[trait_cols].apply(pd.to_numeric, errors="coerce")
#     # pick ID columns if they exist
#     id_cols = [c for c in ["Timestamp", "UUID", "What is your first name?", "What is your last name?"] if c in df.columns]
#     return df[id_cols + trait_cols].copy()


# # def compute_trait_scores_from_ranges(df: pd.DataFrame,
# #                                      trait_ranges: Dict[str, Tuple[int,int]]) -> pd.DataFrame:
# #     df = df.copy()
# #     trait_columns_mapping = {}

# #     for trait, (start, end) in trait_ranges.items():
# #         start_idx = max(0, start-1)
# #         end_idx = min(len(df.columns), end)  # end exclusive
# #         trait_columns_mapping[trait] = list(df.columns[start_idx:end_idx])
# #         if len(trait_columns_mapping[trait]) == 0:
# #             raise ValueError(f"No columns found for trait {trait} with range {start}-{end}")

# #     all_q_cols = [col for cols in trait_columns_mapping.values() for col in cols]
# #     df = ensure_numeric(df, all_q_cols)

# #     for trait, cols in trait_columns_mapping.items():
# #         df[trait] = df[cols].mean(axis=1).round(1)

# #     id_cols = [c for c in ["Timestamp", "What is your first name?", "What is your last name?", "UUID"] if c in df.columns]

# #     return df[id_cols + list(trait_columns_mapping.keys())].copy()


# def determine_strength_growth(user_row: pd.Series, trait_cols: List[str], top_n: int = 3
#                               ) -> Tuple[List[str], List[str]]:
#     """
#     Return strengths (top_n, inclusive of ties) and growth (bottom_n, inclusive of ties).
#     Deterministic tie handling: sort by value desc then by trait name.
#     """
#     traits = user_row[trait_cols].astype(float).copy()
#     # sort descending, tie-breaker alphabetical for determinism
#     sorted_desc = traits.sort_values(ascending=False).sort_index()  # stable deterministic
#     sorted_desc = sorted_desc.sort_values(ascending=False)  # ensures primary sort by value
#     # Strengths (inclusive ties)
#     if len(sorted_desc) == 0:
#         return [], []
#     strengths_cutoff = sorted_desc.iloc[top_n - 1] if top_n <= len(sorted_desc) else sorted_desc.iloc[-1]
#     strengths = sorted_desc[sorted_desc >= strengths_cutoff].index.tolist()

#     # Growths
#     sorted_asc = traits.sort_values(ascending=True).sort_index()
#     sorted_asc = sorted_asc.sort_values(ascending=True)
#     growth_cutoff = sorted_asc.iloc[top_n - 1] if top_n <= len(sorted_asc) else sorted_asc.iloc[-1]
#     growth = sorted_asc[sorted_asc <= growth_cutoff].index.tolist()

#     return strengths, growth

# def aggregate_peer_means(peer_df: pd.DataFrame, trait_cols: List[str], name_col: str="Full Name") -> pd.DataFrame:
#     """
#     Return DataFrame indexed by reviewed person's Full Name with mean trait scores from peers.
#     """
#     # ensure numeric
#     peer_df = ensure_numeric(peer_df, trait_cols)
#     # groupby Full Name and take mean
#     grouped = peer_df.groupby(name_col, as_index=True)[trait_cols].mean().round(1)
#     return grouped

# def compute_consistency(user_row: pd.Series, peer_mean: Optional[pd.Series], trait_cols: List[str],
#                         tolerance: float = 1.0) -> Tuple[float, List[str], List[str]]:
#     if peer_mean is None:
#         return 0.0, [], trait_cols.copy()
#     diffs = (user_row[trait_cols].astype(float) - peer_mean[trait_cols].astype(float)).abs()
#     consistent = diffs[diffs <= tolerance].index.tolist()
#     inconsistent = diffs[diffs > tolerance].index.tolist()
#     consistency_pct = round(len(consistent) / len(trait_cols) * 100, 1)
#     return consistency_pct, consistent, inconsistent

# # -------------------------
# # UI helpers (plot + HTML)
# # -------------------------
# def plot_trait_comparison(user_row: pd.Series, peer_mean_scores: Optional[pd.Series], trait_cols: List[str]):
#     self_scores = [(user_row[trait] / 6) * 100 for trait in trait_cols]
#     peer_scores = [(peer_mean_scores[trait] / 6) * 100 if peer_mean_scores is not None else 0 for trait in trait_cols]
#     delta_scores = [round(peer - self_, 1) for self_, peer in zip(self_scores, peer_scores)]

#     fig = go.Figure()
#     fig.add_trace(go.Bar(y=trait_cols, x=self_scores, name="Self Assessment", orientation="h",
#                          text=[f"{s}%" for s in self_scores], textposition="outside"))
#     fig.add_trace(go.Bar(y=trait_cols, x=peer_scores, name="Peer Review", orientation="h",
#                          text=[f"{p}%" for p in peer_scores], textposition="outside"))
#     for i, trait in enumerate(trait_cols):
#         fig.add_annotation(x=max(self_scores[i], peer_scores[i]) + 5, y=trait,
#                            text=f"Δ {delta_scores[i]}", showarrow=False, xanchor='left')
#     fig.update_layout(barmode='group', title='Self vs Peer Trait Assessment',
#                       xaxis=dict(title='Score (%)', range=[0, 100]),
#                       yaxis=dict(title='Trait'),
#                       height=50 * len(trait_cols) + 150,
#                       margin=dict(l=150, r=50, t=60, b=50))
#     return fig

# def load_css_once(path="styles.css"):
#     try:
#         css = open(path).read()
#         st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
#     except FileNotFoundError:
#         # silently ignore if missing, or log
#         pass

# # -------------------------
# # Streamlit app wiring
# # -------------------------
# def main():
#     st.set_page_config(layout="wide")
#     load_css_once("styles.css")

#     # Use the helper function to get your gspread client
#     client = get_gspread_client(st.secrets)

#     # fetch dataframes (cached)
#     try:
#         data = fetch_sheet_df(client, INDIV_SHEET_NAME)
#         peerdata = fetch_sheet_df(client, PEER_SHEET_NAME)
#     except Exception as e:
#         st.error(f"Error fetching sheets: {e}")
#         st.stop()

#     # Compute trait score frames
#     try:
#         df_traits = compute_trait_scores(data, TRAITS)
#         df_peer_traits = compute_trait_scores(peerdata, TRAITS)
#         st.write("Columns in df_peer_traits:", df_peer_traits.columns.tolist())

#     except KeyError as e:
#         st.error(f"Sheet format problem: {e}")
#         st.stop()

#     # Add full name columns
#     df_traits["Full Name"] = data["What is your first name?"].str.strip() + " " + data["What is your last name?"].str.strip()
#     df_peer_traits["Full Name"] = peerdata["Who are you peer reviewing? (First and Last Name)"].str.strip()


#     # Aggregate peer means safely
#     peer_means_df = aggregate_peer_means(df_peer_traits, TRAITS, name_col="Full Name")

#     # UI: get uuid param or input
#     query_params = st.experimental_get_query_params()
#     uuid_param = query_params.get("uuid", [None])[0]
#     uuid_input = st.text_input("Enter your report code:", value=uuid_param or "")

#     if not uuid_input:
#         st.info("Enter or pass your report code in the URL to view your report.")
#         return

#     # case-insensitive and trimmed lookup
#     uuid_input_stripped = uuid_input.strip()
#     user_rows = data[data.get("UUID", "").astype(str).str.strip() == uuid_input_stripped]

#     if user_rows.empty:
#         st.error("No report found for this report code.")
#         return

#     # Get the user's trait row (from df_traits)
#     user_traits = df_traits[df_traits.get("UUID", "").astype(str).str.strip() == uuid_input_stripped]
#     if user_traits.empty:
#         st.error("No trait data found for this report code.")
#         return

#     user_row = user_traits.iloc[0]

#     # strengths & growth
#     strengths, growth = determine_strength_growth(user_row, TRAITS, top_n=3)

#     # peers: aggregate means by Full Name and find matching mean for this user
#     peer_means_df = aggregate_peer_means(peerdata, TRAITS, name_col="Full Name")
#     # Find peer mean for this user's full name
#     full_name = data.loc[user_rows.index[0], "Full Name"] if "Full Name" in data.columns else None
#     peer_mean_scores = peer_means_df.loc[full_name] if full_name in peer_means_df.index else None

#     consistency_pct, consistent_traits, inconsistent_traits = compute_consistency(user_row, peer_mean_scores, TRAITS)

#     # show a simple dataframe of the user's scores
#     st.dataframe(user_traits[TRAITS].T.rename(columns={user_traits.index[0]: "Individual Score"}))

#     # display message (HTML component)
#     user_name = user_rows["What is your first name?"].iloc[0] if "What is your first name?" in user_rows.columns else "Participant"
#     strengths_str = ", ".join(strengths)
#     growth_str = ", ".join(growth)
#     peer_html = ""
#     if peer_mean_scores is not None:
#         peer_html = f"""
#         <div class="peer-card">
#             <p>Peer consistency: <strong>{consistency_pct}%</strong> ({', '.join(consistent_traits)})</p>
#             <p>Inconsistencies: {', '.join(inconsistent_traits)}</p>
#         </div>
#         """

#     html = f"""
#     <div class="welcome-card">
#       <h2>Welcome, {user_name}, to the Wisdom Playbook 🧭</h2>
#       <div class="congrats-card">
#         <h1>🎉 Congratulations, {user_name}!</h1>
#         <p>Your <strong>strength traits</strong> are: <span class="strengths">{strengths_str}</span>.</p>
#         <p>Your <strong>growth traits</strong> are: <span class="growth">{growth_str}</span>.</p>
#       </div>
#       {peer_html}
#     </div>
#     """
#     components.html(html, height=400)

#     # plot chart
#     fig = plot_trait_comparison(user_row, peer_mean_scores, TRAITS)
#     st.plotly_chart(fig, use_container_width=True)


# if __name__ == "__main__":
#     main()
