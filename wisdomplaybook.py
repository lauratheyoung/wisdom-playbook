#app script
#Backend Logic to handle form data
#UUID needed to be generated per use per form submission --> figure out condition of 1 user 3x friends per form accumulated results can be grouped/related
#Push JSON to streamlit
#Visualise the report in Streamlit
#Email report with pdf attachment of results to user in two stages 1. when they submit their form, 2. when 3 friends complete their forms
#Use admin email

import gspread as gs
import plotly as pl
from google.oauth2.service_account import Credentials
import streamlit as st
import pandas as pd
from urllib.parse import urlparse, parse_qs

#Google sheet setup

scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
creds = Credentials.from_service_account_file("wisdomplaybook.json", scopes=scopes)
client = gs.authorize(creds)

sheet = client.open("https://docs.google.com/spreadsheets/d/1XEIVMPSS69BHDBGPw8fKkuqze5iqXSP7JFfwdGQhSHk/edit?gid=406100282#gid=406100282").Individual

data = pd.Dataframe(sheet.get_all_records())

#Streamlit UI

st.set_page_config(layout="wide")
st.title("Form Submission Report Viewer")

#Get UUI from query params

query_params = st.query_params
uuid_param = query_params.get("uuid", [None])[0]

# Optional: let user input manually if query param not provided
uuid_input = st.text_input("Enter your report UUID:", value=uuid_param or "")

if uuid_input:
    user_data = data[data["UUID"] == uuid_input]
    if not user_data.empty:
        st.success("✅ Report found")
        # TODO: Replace with your designed report
        st.dataframe(user_data)
    else:
        st.error("No report found for this ID.")

