import requests
import streamlit as st


@st.cache_data(show_spinner=False)
def get_answer(msg: str):
    return requests.post("http://localhost:1416/rag_pipeline",
                         json={"input": {"value": msg}, "llm": {"parts": ""}})


message = st.text_input("Question", key="question")

if message:
    with st.spinner("Generating Answer..."):
        response = get_answer(message)

    if response.status_code != 200:
        st.error("There was an error.")
    else:
        response = response.json()
        if not response["llm"]["replies"]:
            st.write("No response...")
        else:
            st.write(response["llm"]["replies"][0])
        st.divider()
        st.divider()
        st.header("Retrieved Documents for Generated Answer")
        for document in response["documents_output"]["documents"]:
            st.divider()
            st.subheader(document["meta"]["title"])
            st.markdown(f"### Guest: {document['meta']['guest']}")
            st.write("<pre> " + document["content"] + " </pre>", unsafe_allow_html=True)
