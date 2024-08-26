import pandas as pd
from prophet import Prophet
from ibm_watsonx_ai.foundation_models.prompts import PromptTemplateManager, PromptTemplate
import logging
from ibm_watsonx_ai import APIClient
from ibm_watsonx_ai import Credentials
from ibm_watsonx_ai.foundation_models.utils.enums import ModelTypes
from ibm_watsonx_ai.foundation_models import ModelInference
from ibm_watsonx_ai.metanames import GenTextParamsMetaNames as GenParams
from ibm_watsonx_ai.foundation_models.utils.enums import DecodingMethods
import requests
import json
from flask import Flask, request, jsonify, make_response
from flask_cors import CORS, cross_origin
from dotenv import load_dotenv
import io


app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}}, supports_credentials=True)
load_dotenv()



# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)


# IBM Watson credentials
credentials = Credentials(
    url='',
    api_key=''
)


project_id =''



prompt_mgr = PromptTemplateManager(credentials=credentials, project_id=project_id)

# Create and store the prompt template
prompt_template = PromptTemplate(
    name="Agronomist Assistant",
    model_id=ModelTypes.FLAN_T5_XXL,
    model_params={GenParams.DECODING_METHOD: "sample"},
    description="This model assists with agricultural inquiries.",
    task_ids=["generation"],
    input_variables=["object"],
    instruction="Provide a detailed explanation related to the following agricultural concept.",
    input_prefix="Farmer",
    output_prefix="Agronomist Assistant",
    input_text="What is {object} and how does it benefit crop production?",
    examples=[["What is crop rotation and how does it benefit crop production?",
               "Crop rotation is the practice of growing different types of crops in the same area across a sequence of seasons. It helps improve soil health, reduce pest and disease risks, and increase crop yield."]]
)


stored_prompt_template = prompt_mgr.store_prompt(prompt_template=prompt_template)

client = APIClient(credentials)
client.set.default_project(project_id)

parameters = {
    GenParams.DECODING_METHOD: DecodingMethods.GREEDY,
    GenParams.MIN_NEW_TOKENS: 1,
    GenParams.MAX_NEW_TOKENS: 1000,
    #GenParams.STOP_SEQUENCES: ["\n\n\n"]
}

model_id = ModelTypes.GRANITE_13B_CHAT_V2

model = ModelInference(
    model_id=model_id,
    params=parameters,
    credentials=credentials,
    project_id = project_id
    )

@app.route('/hello', methods=['GET'])
def hello():
    return "Hello"


@app.route('/test_ai', methods=['POST'])
def test_ai():
    data = request.json
    prompt = data.get('prompt')

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        # Use the stored prompt template to generate a response
        response = model.generate_text(prompt="hi", params=parameters)

        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



@app.route('/forecast', methods=['POST', 'OPTIONS'])
@cross_origin()
def forecast():
    if request.method == "OPTIONS":
        response = jsonify({'message': 'OK'})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST")
        return response

    try:
        logger.debug("Received POST request to /forecast")
        data = request.json
        logger.debug(f"Request data: {data}")

        csv_data = data.get('data')
        ai_prompt = data.get('prompt')

        if not csv_data:
            logger.error("No CSV data provided")
            return jsonify({"error": "No CSV data provided"}), 400

        # Convert the CSV data string to a pandas DataFrame
        df = pd.read_csv(io.StringIO(csv_data))
        logger.debug(f"DataFrame head: {df.head()}")

        # Ensure the dataframe has the correct columns for Prophet
        if 'ds' not in df.columns or 'y' not in df.columns:
            logger.error("CSV data missing required columns")
            return jsonify({"error": "CSV data must have 'ds' and 'y' columns"}), 400

        # Convert 'ds' column to datetime if it's not already
        df['ds'] = pd.to_datetime(df['ds'])

        # Initialize the Prophet model and fit it to the data
        logger.debug("Initializing and fitting Prophet model")
        model = Prophet()
        model.fit(df)

        # Create a dataframe to hold future dates for prediction
        future = model.make_future_dataframe(periods=14)
        logger.debug(f"Future dataframe head: {future.head()}")

        # Make predictions
        logger.debug("Making predictions")
        forecast = model.predict(future)
        logger.debug(f"Forecast dataframe head: {forecast.head()}")

        # Prepare forecast data
        forecast_data = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_dict(orient='records')

        # Convert datetime objects to strings
        for item in forecast_data:
            item['ds'] = item['ds'].strftime('%Y-%m-%d %H:%M:%S')

        # Prepare a summary of the forecast for AI explanation
        forecast_summary = forecast_data[:5]  # Take first 5 forecasts for summary

        try:
            forecast_summary_str = json.dumps(forecast_summary, indent=2)
        except TypeError as e:
            logger.error(f"JSON serialization error: {e}")
            # If serialization fails, try a more robust approach
            forecast_summary_str = json.dumps([{k: str(v) for k, v in item.items()} for item in forecast_summary],
                                              indent=2)

        # Prepare the prompt for AI
        ai_prompt_full = f"{ai_prompt}\n\nHere's a summary of the forecast data:\n{forecast_summary_str}\n\nPlease provide an explanation and insights based on this data."

        # Get AI explanation (placeholder for actual implementation)
        # ai_explanation = "AI explanation placeholder"  # Replace with actual AI generation

        model_id = ModelTypes.GRANITE_13B_CHAT_V2

        modelai = ModelInference(
            model_id=model_id,
            params=parameters,
            credentials=credentials,
            project_id=project_id
        )
        ai_explanation = modelai.generate_text(prompt=ai_prompt_full, params=parameters)

        # Prepare the response
        response_data = {
            "forecast": forecast_data,
            "ai_explanation": ai_explanation
        }

        logger.debug("Preparing to send response")
        return jsonify(response_data), 200

    except Exception as e:
        logger.exception(f"An error occurred: {str(e)}")
        return jsonify({"error": str(e)}), 500

@app.route('/my_farm/crop_planning', methods=['POST'])
def crop_planning():
    data = request.json
    farm_data = data.get('farm_data')
    prompt = f"""
    As an AI farm management assistant, analyze the following farm data and provide 
    crop planning recommendations for the next season. Consider factors such as soil type, 
    climate, market demand, and crop rotation. Farm data: {farm_data}
    """
    try:
        response = model.generate_text(prompt=prompt, params=parameters)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/financial_analysis/cash_flow_forecast', methods=['POST'])
def cash_flow_forecast():
    data = request.json
    financial_data = data.get('financial_data')
    prompt = f"""
    As a financial analyst for a farm, review the following financial data and provide 
    a cash flow forecast for the next 12 months. Include potential risks and opportunities. 
    Financial data: {financial_data}
    """
    try:
        response = model.generate_text(prompt=prompt, params=parameters)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/disease_analysis/early_detection', methods=['POST'])
def disease_early_detection():
    data = request.json
    crop_data = data.get('crop_data')
    prompt = f"""
    As a plant pathologist, analyze the following crop data for any early signs of disease. 
    Provide potential diagnoses, risk levels, and recommended actions. Crop data: {crop_data}
    """
    try:
        response = model.generate_text(prompt=prompt, params=parameters)
        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/twitter_ai', methods=['POST'])
def twitter_ai():
    data = request.json
    prompt = data.get('prompt')

    #prompt_input = prompt

    prompt_txt = """
       You are a social media expert for a farm. Your task is to create an engaging Twitter post about {topic}. 
       The post should be informative, catchy, and appropriate for a farming audience. 
       Include relevant hashtags and emojis to increase engagement.

       Rules:
       1. Keep the post under 280 characters (including hashtags and emojis).
       2. Use 2-3 relevant hashtags.
       3. Include 1-2 appropriate emojis.
       4. Make the content relatable to farmers and agriculture enthusiasts.
       5. If applicable, mention sustainable farming practices or seasonal information.

       Example:
       Input: Harvest season
       Output: 🌾 It's harvest time! Our fields are golden and ready for gathering. Hard work pays off! 🚜 #HarvestSeason #FarmLife #Agriculture

       """

    prompt_input = prompt_txt + ' ' + prompt

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        # Use the stored prompt template to generate a response
        response = model.generate_text(prompt=prompt_input, params=parameters)

        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route('/linkedin_ai', methods=['POST'])
def linkedin_ai():
    data = request.json
    prompt = data.get('prompt')

    prompt_txt = """
    You are a social media expert for a farm. Your task is to create engaging LinkedIn posts about {topic}. 
    The posts should be informative, professional, and appropriate for a farming and business audience. 

    Rules:
    1. Keep the posts concise and clear.
    2. Use industry-related terminology.
    3. Make the content relatable to farmers, agriculture enthusiasts, and business professionals.
    4. If applicable, mention sustainable farming practices or seasonal information.
    5. Include 1-2 relevant emojis to add a touch of personality to the posts.
    6. Generate multiple post options.

    Example:
    Input: Harvest season
    Output: 
    Option 1: "🌾 It's harvest time! Our fields are golden and ready for gathering. This is the result of months of hard work and careful planning. 🚜"
    Option 2: "🍂 Harvest season is here. It's rewarding to see our sustainable farming practices pay off with a bountiful yield. 🌽"
    Option 3: "🌅 As we begin harvest season, we're reminded of the importance of agriculture in our economy and our daily lives. 🌍"
    """

    prompt_input = prompt_txt + ' ' + prompt

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        # Use the stored prompt template to generate a response
        response = model.generate_text(prompt=prompt_input, params=parameters)

        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/finance_ai', methods=['POST'])
def finance_ai():
    data = request.json
    prompt = data.get('prompt')
    csvdata = data.get('data')
    print(csvdata)

    prompt_txt = f"""
    Process this, do finanical calculations using metrics and ratios and write a good report with emojis for the following financial data for the company and suggest any chnages and improvemnets that can be done you are part of the farm as a CFO. Data={csvdata}
    """

    prompt_input = prompt_txt + ' ' + prompt

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        # Use the stored prompt template to generate a response
        response = model.generate_text(prompt=prompt_input, params=parameters)

        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/customer_ai', methods=['POST'])
def customer_ai():
    data = request.json
    prompt = data.get('prompt')
    csvdata = data.get('data')
    print(csvdata)

    prompt_txt = f"""
    Process this, do calculations using metrics and ratios and write a good report with emojis for the following Customer data for the company and suggest any chnages and improvemnets that can be done you are part of the company as a CMO, Data = {csvdata}
    """

    prompt_input = prompt_txt + ' ' + prompt

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        # Use the stored prompt template to generate a response
        response = model.generate_text(prompt=prompt_input, params=parameters)

        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/feedback_ai', methods=['POST'])
def feedback_ai():
    data = request.json
    prompt = data.get('prompt')
    csvdata = data.get('data')
    print(csvdata)
    print(prompt)

    prompt_txt = f"""
    Process and write a good report from this feedback data as the Chief Growth officer.  give insights and predictions for next week and where the farm should put focus. Data {csvdata}
    """

    prompt_input = prompt_txt + ' ' + prompt

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        # Use the stored prompt template to generate a response
        response = model.generate_text(prompt=prompt_input, params=parameters)

        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/disease_ai', methods=['POST'])
def disease_ai():
    data = request.json
    prompt = data.get('prompt')
    csvdata = data.get('data')
    print(csvdata)
    print(prompt)

    prompt_txt = f"""
    You are a farm doctor who gets descriptions of problems and says what they are, what the cause could be and what the next steps should be  Data {csvdata}
    """

    prompt_input = prompt_txt + ' ' + prompt

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        # Use the stored prompt template to generate a response
        response = model.generate_text(prompt=prompt_input, params=parameters)

        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/agronomist_ai', methods=['POST', 'OPTIONS'])
@cross_origin()
def agronomist_ai():
    if request.method == "OPTIONS":
        response = jsonify({'message': 'OK'})
        response.headers.add("Access-Control-Allow-Origin", "*")
        response.headers.add("Access-Control-Allow-Headers", "Content-Type")
        response.headers.add("Access-Control-Allow-Methods", "POST")
        return response

    data = request.json
    prompt = data.get('prompt')
    lat = data.get('lat')
    lon = data.get('lon')
    apiKey = '';

    print(prompt)

    response = requests.get(f'https://api.isda-africa.com/v1/soilproperty?lat={lat}&lon={lon}&key={apiKey}')
    soil_data = response.json()

    prompt_txt = f"""
    You are an agronomist for a farm. You receive descriptions of problems and provide advice on what they are, what the cause could be, and what the next steps should be. 
    Soil Data: {soil_data}
    """

    prompt_input = prompt_txt + ' ' + prompt

    if not prompt:
        return jsonify({"error": "No prompt provided"}), 400

    try:
        # Use the stored prompt template to generate a response
        response = model.generate_text(prompt=prompt_input, params=parameters)

        return jsonify({"response": response})
    except Exception as e:
        return jsonify({"error": str(e)}), 500



if __name__ == '__main__':
    print(f"Asset id: {stored_prompt_template.prompt_id}")
    print(f"Is it a template?: {stored_prompt_template.is_template}")
    app.run(host='0.0.0.0', port=8000)