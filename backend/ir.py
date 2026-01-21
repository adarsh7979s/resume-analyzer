from google import genai

query = "Who is iron man?"

def ai(query):

    # Configure the API key
    client = genai.Client(api_key="AIzaSyDK_pd-erHvGXTm3-o2humvv2aruwCZgo0")

    # Define the query
    user_input = query.lower() + " (Please give me very short response.)"
       
    try:
        # Initialize the model
        response = client.models.generate_content(
           model="gemini-2.5-flash", contents = user_input
        )
        print(response.text)
        return response
        
    
    except Exception as e:
        print("An error occurred")
        return None
    


ai(query)