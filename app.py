import gradio as gr

from model import predict_sentiment

GITHUB_LINK = "https://github.com/samiali12/twitter-sentiment-analysis"
COFFEE_LINK = "https://www.buymeacoffee.com/samiali"  # <-- replace with your BuyMeACoffee link


with gr.Blocks(theme=gr.themes.Soft()) as app:
    title="Twitter Sentiment Analysis",
    gr.Markdown(
        """
        # 🌟 Twitter Sentiment Analysis  
        Enter a tweet below and find out if it's **Positive** or **Negative**.  
        _Model: Naive Bayes trained on NLTK Twitter samples_
        """
    ),
    with gr.Row():
        with gr.Column():
            text = gr.Textbox(
                placeholder="Type your tweet here...",
                lines=3,
                label="Your Tweet"
            ),
            btn = gr.Button("🔍 Analyze Sentiment", variant="primary")
        with gr.Column():
            output = gr.Label(label="Prediction")

    gr.Markdown(
        f"""
        🔗 **Source Code on [GitHub]({GITHUB_LINK})**  

        ☕ If you like this project, consider [buying me a coffee]({COFFEE_LINK})  

        <a href="{COFFEE_LINK}" target="_blank">
          <img src="https://cdn.buymeacoffee.com/buttons/default-orange.png" 
               alt="Buy Me A Coffee" height="41" width="174">
        </a>
        """
    )

    gr.Markdown("💾 All predictions are stored for analysis.")
    btn.click(predict_sentiment, inputs=text, outputs=output)

if __name__ == '__main__':
    app.launch(share=True)