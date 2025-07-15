from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os
from random import choice
import re
import dateparser
import requests
import datetime
import spacy
import json



# Loading the training data from the JSON file
with open("intent_classifier_resources_ai/training_data_bookings_inquiry.json", "r") as f:
    training_data_bookings_inquiry = json.load(f)

    # The following json file stores all the training data for each category.
    # There are multitudes of examples for each category:
    #   1. Focusing on switching the order of words when giving responses.
    #   2. Using diverse choices of vocabulary and possible synonyms when entering 
    #      the training data to optimize different possibilities of responses.
    #   3. Ensuring that each category has at least 30 different possible ways 
    #      of responding to a query. (At the time of writing this comment, I hope 
    #      this number will increase drastically as I aim to gather as much 
    #      training data as possible to train my model.)
    #   4. Focusing on training data that uses category-specific keywords, 
    #      which makes it easier for the model to distinguish between categories 
    #      and correctly identify what the user needs.
    #   5. Adding categories that consist of "off-topic" responses, so the model 
    #      can learn to distinguish between real and fake requests.


# Loading the dialogue/prompts flows from the JSON file
with open("intent_classifier_resources_ai/dialogue_flows.json", "r") as f:
    dialogue_flows = json.load(f)


# 2. Flatten the data into two lists
training_sentences = []
training_labels = []

for label, sentences in training_data_bookings_inquiry.items():
    training_sentences.extend(sentences)
    training_labels.extend([label] * len(sentences))

# 3. Vectorize the text
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(training_sentences)

# 4. Train the model
model = MultinomialNB()
model.fit(X_train, training_labels)

# Save model and vectorizer for session memory
with open("intent_classifier_resources_ai/model.pkl", "wb") as f: pickle.dump(model, f)
with open("intent_classifier_resources_ai/vectorizer.pkl", "wb") as f: pickle.dump(vectorizer, f)


conversation_context = {
    "last_intent": None,
    "step": 0,
    "params": {},
    "history": []
}


TREATMENT_DURATIONS = {
    
    # Specific durations for each dental service are measured in minutes.
    # Depending on the service selected by the user, the duration will vary.
    # For example:
    #   - Extractions may take around 90 minutes.
    #   - Braces consultations typically take less time, around 30 minutes.
    # This information is useful for scheduling appointments accurately
    # and avoiding overlaps in the calendar.

    "book_whitening": 60,
    "book_cleaning": 60,
    "book_filling": 60,
    "book_extraction": 90,
    "book_root_canal": 90,
    "book_braces": 30,
}


def predict_intent(user_input):


    # Using the training data above (with all the different categories), the model
    # will be trained using the Naive Bayes method in combination with vectorization.
    # This allows the model to estimate which category a user's message most closely matches,
    # assigning a confidence score to each possible category.
    #
    # For now, the model selects the category with the highest confidence score.
    # That selected category is then passed to later functions such as `extract_slot`
    # and `handle_response` to continue the conversation flow.


    with open("intent_classifier_resources_ai/model.pkl", "rb") as f: model = pickle.load(f)
    with open("intent_classifier_resources_ai/vectorizer.pkl", "rb") as f: vectorizer = pickle.load(f)
    X_test = vectorizer.transform([user_input])
    prediction = model.predict(X_test)[0]
    confidence = model.predict_proba(X_test).max()
    return prediction, confidence



def extract_slot(user_input): 
    
    # The extract_slot function's main purpose is to parse out the important information that the user gives
    # in regards to their full name, date, time etc. This information is vital for it to be later stored
    # into the SQLite data base with the FASTapi (for it to be seen only in the dentists database and for 
    # appointmnets to be much more easily handled).

    slots = {} # initializing this dictionary where i can store the individaul pieces of information for the person booking the appointmnet (there full name, date, time, etc.). 

    # This specfic section is meant only to parse the name using a pre-trained model that has been imported
    
    nlp = spacy.load("en_core_web_sm") # Loads the model
    doc = nlp(user_input) # Reads the input
    for ent in doc.ents: # Parses the full name (downside is that this pre-trained is only good at parsing common english names. I would need to train my own model for it to detect names of other langugues)
        if ent.label_ == "PERSON":
            slots["Full name: "] = ent.text # Adds it to the dictionary upon parsing the full name. Note that there is the occasional misidentification of name, and this ussually occurs for less common names, or names that do not have English origin.

    # Using the python librabry "dateparser" which can directly parse the three things that we are looking for which is
    # the day of the week, the exact month/day/year and the time of the day. 
    

    cleaned_input = re.sub(r"\b([A-Za-z]+)'s\b", r"\1", user_input) # removes possessive endings: ex. Friday's, Tuesday's etc.
    cleaned_input = re.sub(r"\b([A-Za-z]+)s\b", r"\1", cleaned_input) # removes plural
    cleaned_input = re.sub(r"[^A-Za-z0-9\s:]", "", cleaned_input) # removinf punctuation like commas etc.


    # the second parameter/argumnet of this function is meant that if a person for example says 
    # they want an appoinment on monday. It will look at the closest following Monday (not look at the 
    # Monday which had already occurred since its trying to book for a future date).

    parsed_date = dateparser.parse(cleaned_input, settings={"PREFER_DATES_FROM": "future"})

    if parsed_date:
        if parsed_date.strftime('%A'): # Stores day of the week (ex. Monday, Tuesday, etc.)
            slots["Day of the Week: "] = parsed_date.strftime('%A') 

        if parsed_date.strftime('%Y-%m-%d'): # Stores date (ex. January 3rd, 2025)
            slots["Date: "] = parsed_date.strftime('%Y-%m-%d')

        if parsed_date.strftime('%H:%M'): # Stores time (ex. 18:00PM)
            slots["Time: "] = parsed_date.strftime('%H:%M')
    
    # If the dateparser librabry fails, we'll use this function to re-parse the weekday. 
    # I've noticed that the dataparser librabry in Python is not as accurate as I thought (it misses a lot of input that I give it, 
    # escpecially when the weekday is given along side many words around it in a vague context.)
    
    if not parsed_date:
        weekdays = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        for i, day in enumerate(weekdays):
            if day in cleaned_input.lower():
                today = datetime.date.today()
                today_index = today.weekday()
                delta_days = (i - today_index + 7) % 7 or 7 # Essentially checks for the following day in terms of next week ex. Next Monday, Next Friday etc.
                next_day = today + datetime.timedelta(days=delta_days)
                parsed_date = datetime.datetime.combine(next_day, datetime.time())
                slots["Day of the Week: "] = parsed_date.strftime('%A') # Stores the weekday back given that teh orginal dateparser librabry fails and returns nothing.
                break


    # Extracting time using regex if dateparser fails to extract it. Similar to the above reason,
    # sometimes the dateparser does not detect timings very well, so gthis back-up function checks for the correct timing. 

    time_match = re.search(r'(\d{1,2})(?:[:.](\d{2}))?\s*(am|pm)?', cleaned_input, re.IGNORECASE)
    if time_match:
        hour = int(time_match.group(1))
        minute = int(time_match.group(2)) if time_match.group(2) else 0
        meridian = time_match.group(3)

        if meridian: # Converting the time to the 24hour clock since it would be easier to communicate this.
            if meridian.lower() == 'pm' and hour < 12:
                hour += 12
            elif meridian.lower() == 'am' and hour == 12:
                hour = 0
        else: # this is for cases if a person says for exmaple: "I can come in after 3".
            # No AM/PM mentioned — assume PM if the hour is reasonable (1–11)
            if 1 <= hour <= 11:
                hour += 12  # default to PM

        # Only accept times between 09:00 and 17:00 (opening hours for the dentist)
        if 9 <= hour < 17:
            parsed_time = f"{hour:02d}:{minute:02d}"
            slots["Time: "] = parsed_time
        else:
            print("Sorry, our clinic is only open between 9:00 AM and 5:00 PM.")
        


    return slots



def handle_response(user_input):
    context = conversation_context
    context["history"].append(user_input)

    # Step 1: Predict intent and extract new slots
    predicted_intent, confidence = predict_intent(user_input)
    found_slots = extract_slot(user_input)

    # Step 2: FSM setup
    current_intent = context.get("last_intent")
    step = context.get("step", 0)

    if step == 0 and not current_intent:
        # First-time intent detection
        context["last_intent"] = predicted_intent
        context["params"] = found_slots
        flow = dialogue_flows.get(predicted_intent, [])
        context["step"] = 0
    else:
        # Continue with previous intent
        predicted_intent = current_intent
        flow = dialogue_flows.get(predicted_intent, [])
        context["params"].update(found_slots)

    print(f"Intent: {predicted_intent}, Step: {context['step']}, Slots: {context['params']}")

    # Step 3: Check for missing required slots
    required_slots = [step["expect"] for step in flow if step["expect"] != "end"]
    missing = [
        slot for slot in required_slots
        if slot not in context["params"] or not context["params"][slot]
    ]

    # Step 4: Booking logic (only if all required slots are filled and FSM is done)
    if not missing and context["step"] >= len(flow):
        full_name = context["params"].get("Full name: ")
        day_of_week = context["params"].get("Day of the Week: ")
        date = context["params"].get("Date: ")
        time_str = context["params"].get("Time: ")

        treatment = predicted_intent.replace("book_", "").replace("_", " ").title()
        duration = TREATMENT_DURATIONS.get(predicted_intent, 60)

        payload = {
            "name": full_name,
            "date": date,
            "time": time_str,
            "treatment": treatment,
            "duration": duration
        }

        try:
            response = requests.post("http://127.0.0.1:8000/api/add_appointment", json=payload)
            confirmation = "You're booked!" if response.status_code == 200 else "Booking failed."
        except Exception as e:
            confirmation = f"Could not connect to the server: {e}"

        parsed = (
            f"[Parsed Info] Name: {full_name}, Day: {day_of_week}, Date: {date}, "
            f"Time: {time_str}, Treatment: {treatment}, Duration: {duration} mins"
        )

        # Reset FSM context
        context["step"] = 0
        context["params"] = {}
        context["last_intent"] = None

        return confirmation + "\n" + parsed + "\nIs there anything else I can help you with?"

    # Step 5: FSM prompt to collect info (only advance if value is filled)
    expected_slot = flow[context["step"]]["expect"]
    if expected_slot in context["params"] and context["params"][expected_slot]:
        context["step"] += 1
        return handle_response(user_input)  # recursively advance
    else:
        return flow[context["step"]]["prompt"]



# 7. Command line chat loop
if __name__ == "__main__":
    print("Welcome to the Dental Assistant AI with FSM! Type 'exit' to quit.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "exit":
            print("Goodbye!")
            break
        response = handle_response(user_input)
        print("Assistant:", response, "\n")

