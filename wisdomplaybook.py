import gspread as gs
import plotly as pl
from google.oauth2.service_account import Credentials
import streamlit as st
import pandas as pd
from urllib.parse import urlparse, parse_qs

#Google sheet setup
creds_info = st.secrets

print(creds_info)


scopes = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
client = gs.authorize(creds)

# Test connection
st.write("Connected sheets:", [s.title for s in client.openall()])

# Open sheets
sheet = client.open_by_url("https://docs.google.com/spreadsheets/d/1XEIVMPSS69BHDBGPw8fKkuqze5iqXSP7JFfwdGQhSHk/edit#gid=406100282").worksheet("Form_Responses2")

#Load into dataframe
data = pd.DataFrame(sheet.get_all_records())

#Display first rows
st.dataframe(data.head())


#Streamlit UI
st.set_page_config(layout="wide")
st.title("Form Submission Report Viewer")
st.title("Hello Streamlit!")
st.write("If you see this, Streamlit is working.")

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

# test update
