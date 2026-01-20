import streamlit as st

def check_password():
    """
    Returns `True` if the user had the correct password.
    Stops execution via st.stop() if password is incorrect.
    """
    
    # 1. Dev/Debug Mode Bypass
    # We use .get() to return False if 'dev_mode' is missing from secrets
    if st.secrets.get("dev_mode", False):
        st.toast("🔓 Dev Mode Active", icon="🛠️")
        return True

    # 2. Initialize Session State
    if "password_correct" not in st.session_state:
        st.session_state.password_correct = False

    # 3. Check if already authenticated
    if st.session_state.password_correct:
        return True

    # 4. Show Input
    st.text_input(
        "Please enter the app password", 
        type="password", 
        key="password_input"
    )
    
    # 5. Validate Input
    if st.session_state.get("password_input"):
        input_pwd = st.session_state.password_input
        # Compare with secrets
        if input_pwd == st.secrets["app_password"]:
            st.session_state.password_correct = True
            st.rerun() # Refresh to show app
        else:
            st.error("😕 Password incorrect")
            st.stop()
    
    # Stop if we haven't authenticated yet (and just showed the input)
    st.stop()
