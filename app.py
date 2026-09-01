from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from agent.agentic_workflow import GraphBuilder
from utils.save_to_doc import save_document
import os
import datetime
from dotenv import load_dotenv
load_dotenv()

app = Flask(__name__)
CORS(app)  # Allow cross-origin requests

@app.route("/")
def home():
    return render_template("index.html")  # Serve HTML page

@app.route("/query", methods=["POST"])
def query_travel_agent():
    try:
        data = request.get_json()
        if not data or "question" not in data:
            return jsonify({"error": "Missing 'question' field"}), 400

        question = data["question"]
        print(f"Received question: {question}")

        graph = GraphBuilder(model_provider="groq")
        react_app = graph()

        # Save graph as PNG
        # png_graph = react_app.get_graph().draw_mermaid_png()
        # with open("my_graph.png", "wb") as f:
        #     f.write(png_graph)
        # print(f"Graph saved as 'my_graph.png' in {os.getcwd()}")

        messages = {"messages": [question]}
        output = react_app.invoke(messages)

        if isinstance(output, dict) and "messages" in output:
            final_output = output["messages"][-1].content
        else:
            final_output = str(output)

        return jsonify({"answer": final_output})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
