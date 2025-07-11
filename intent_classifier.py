from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
import pickle
import os
from random import choice
import re
import dateparser
import requests
import datetime




# 1. Structured training data
training_data = {
    "book_cleaning": [
    "I need to book a teeth cleaning",
    "Please schedule my dental cleaning",
    "I want to set up a cleaning appointment",
    "I'd like to schedule a cleaning visit",
    "Can you book me for a cleaning?",
    "Time to clean my teeth, book me in",
    "Help me get a cleaning appointment",
    "I’m looking to book a cleaning",
    "Book a visit for tooth cleaning",
    "When’s the next available cleaning slot?",
    "I need to come in for a cleaning",
    "Please help me schedule my teeth cleaning",
    "Can I arrange a cleaning session?",
    "How do I book a cleaning?",
    "Put me on the list for a cleaning",
    "I'd like a cleaning soon",
    "Book a professional cleaning for me",
    "Please add me for a cleaning",
    "I want to get my teeth cleaned this week",
    "Book me in for a hygiene appointment",
    "I’d like a basic cleaning",
    "I need to get plaque removed",
    "How can I book a cleaning session?",
    "I want a dentist cleaning slot",
    "Can I book an appointment for plaque removal?",
    "Schedule a hygiene cleaning please",
    "Reserve a spot for my dental cleaning",
    "I want to clean my teeth at the clinic",
    "I'd like to schedule oral cleaning",
    "Can you arrange a cleaning for me?",
    "Please put me down for a teeth cleaning",
    "Is there a cleaning appointment available?",
    "I’m due for a hygiene cleaning",
    "I'd like to book an appointment to clean my teeth",
    "Add me to the schedule for a cleaning",
    "I want to remove buildup on my teeth",
    "Need an appointment for plaque cleanup",
    "Time to clean up my teeth, can I book?",
    "I'd like to schedule a hygiene visit",
    "Book me a slot for professional cleaning",
    "Can I get in for a cleaning this week?",
    "I need to get my dental cleaning done",
    "Teeth cleaning booking please",
    "Clean my teeth — book me in",
    "I'd like to visit for oral hygiene",
    "Can I get my cleaning soon?",
    "Please schedule me a quick cleaning",
    "I'd love to book a cleaning",
    "Let’s book a teeth cleaning session"
],
   "teeth_whitening": [
    "I'd like to schedule a whitening treatment",
    "I need to book a whitening session",
    "Book a time to whiten my teeth",
    "Please set an appointment for whitening",
    "Can I come in for a whitening service?",
    "I want to book a teeth whitening slot",
    "I need an appointment for whitening",
    "I’m looking to whiten my teeth",
    "I'd like to come in for whitening",
    "Can you help me book whitening?",
    "I want a whitening consultation",
    "Book my whitening treatment",
    "Put me down for teeth whitening",
    "I want to whiten my teeth professionally",
    "Can I reserve a whitening session?",
    "I'd like to do a whitening procedure",
    "Please help me set up a whitening",
    "Teeth whitening appointment, please",
    "I'd like an appointment to brighten my teeth",
    "Book me for a smile whitening",
    "I need to remove stains from my teeth",
    "Can I get a whitening consult?",
    "I'd like to book a cosmetic whitening service",
    "I'd like a whitening procedure scheduled",
    "I want brighter teeth, book me in",
    "Add me to the whitening schedule",
    "Whitening service booking, please",
    "Help me whiten my smile",
    "I want to brighten my teeth with you",
    "Sign me up to whiten my teeth",
    "I'd love to get a teeth whitening session",
    "I’m interested in scheduling teeth whitening",
    "Book whitening treatment slot please",
    "Set a time for teeth whitening",
    "I'd like to start with a whitening procedure",
    "I want your whitening service",
    "Book me for stain removal session",
    "I want a teeth whitening visit",
    "Do you do teeth whitening",
    "I want my teeth to be whiter",
    "I want whiter teeth",
    "My teeth are yellowing, I need to whiten my teeth",
    "Hi, I’d like to book a whitening session",
    "Can I whiten my teeth?",
    "I need to get teeth whitening",
    "I want a whitening session",
    "My teeth need whitening",
    "Book me for teeth whitening",
    "Whitening appointment please",
    "I’d like to get my teeth whitened",
    "I need to whiten my teeth",
    "I’m looking to brighten my smile",
    "Teeth whitening this week",
    "Can I schedule teeth whitening?",
    "Set up a whitening treatment",
    "Please book me for whitening",
    "Whitening on Friday",
    "I want to get my teeth whitened"
],
   "cancel_appointment": [
    "I want to cancel my visit",
    "Please cancel my appointment",
    "I can't come anymore, cancel it",
    "Can I cancel my slot?",
    "Cancel my visit to the dentist",
    "I’m no longer available, cancel the booking",
    "I have to cancel the appointment",
    "I won't be able to make it, cancel please",
    "Just cancel my appointment",
    "Can you take me off the schedule?",
    "Cancel my dental visit",
    "I don’t need the appointment anymore",
    "Please remove me from the schedule",
    "I want to cancel my slot",
    "I changed my mind, please cancel",
    "Don’t need the appointment now, cancel it",
    "Please cancel my reservation",
    "I want to cancel my time slot",
    "I’ll skip the appointment, cancel it",
    "Can I stop my appointment?",
    "No longer attending, please cancel",
    "Cancel the dental checkup I booked",
    "I can’t make it, cancel my visit",
    "Remove my name from the list",
    "I’ve decided to cancel",
    "Can you drop my appointment?",
    "I want to call off my appointment",
    "Can I undo my booking?",
    "Cancel my cleaning",
    "I’m cancelling my appointment",
    "I won't show up, cancel it",
    "Can you erase my booking?",
    "Forget the appointment, cancel it",
    "Don’t book me anymore",
    "Cancel my spot at the clinic",
    "No longer need the visit, cancel",
    "Call off my dental appointment",
    "Remove my cleaning from the schedule",
    "Please delete my dentist visit",
    "Take me off your schedule",
    "I’m not coming anymore, cancel",
    "Don’t count me in for the appointment",
    "Can you drop me from the list?",
    "I'm unavailable, cancel my slot",
    "Forget about my appointment"
],
    "reschedule_appointment": [
    "Can I change my appointment?",
    "I'd like to reschedule my visit",
    "Is it possible to push my appointment?",
    "Please help me reschedule",
    "Can I adjust my booking?",
    "I won’t make it, can we reschedule?",
    "I need to move the date of my appointment",
    "Can I come another time instead?",
    "I can’t make the original time, need to reschedule",
    "I want to reschedule to a more convenient time",
    "My plans changed, I need a new slot",
    "I’m busy, can I get a new time?",
    "Can we shift the time of my cleaning?",
    "I need to postpone my appointment",
    "Something came up, can I move my dental visit?",
    "I want to change the day of my appointment",
    "I can't come at that time, let’s rebook",
    "Can you help me reschedule for a later time?",
    "Let’s set my appointment for another time",
    "I’m running late, can we reschedule?",
    "I’ll be unavailable, move my appointment please",
    "Can we change my cleaning to a different day?",
    "Please give me a new time for my checkup",
    "Can I get another slot for the same appointment?",
    "I missed my appointment, can we move it?",
    "I had an emergency, need to reschedule",
    "Can you push my appointment to later?",
    "Please update the appointment time",
    "Can you cancel and rebook it for later?",
    "I need to switch my appointment date",
    "Can I do it another day?",
    "I need to find a different time for my appointment",
    "Please update my visit to another time",
    "Can you help me shift it to a later time?",
    "Is there a better time for me?",
    "I want to choose another appointment day",
    "Let’s reschedule for a better time",
    "Please book me for a different time",
    "I can’t do that time, change it please",
    "I want to move my checkup to a different date",
    "Can you give me a later appointment?",
    "Can we move the dental visit to another time?"
],
   "book_checkup": [
    "Can I come in for a routine checkup?",
    "I’d like to get my teeth checked",
    "Can you book me for a regular checkup?",
    "Please set up a dental checkup for me",
    "I think it's time for a checkup",
    "I want to see the dentist for a quick check",
    "I’d like to come in for a dental exam",
    "Can I book an appointment for a general checkup?",
    "Help me book a dental evaluation",
    "Schedule a regular dentist visit for me",
    "I need an oral health checkup",
    "Can you add me for a dental check?",
    "I need my teeth looked at",
    "Please book me a regular dental check",
    "I want to check the condition of my teeth",
    "Let’s book a dentist exam",
    "I want to book a follow-up checkup",
    "I’m looking for a basic checkup",
    "Book my dentist checkup",
    "Please schedule me for an exam",
    "I’d like to make a checkup appointment",
    "Set me up with a teeth check appointment",
    "How do I book a routine dental check?",
    "Book me for a dentist visit",
    "Can I have a quick oral exam?",
    "I just want to get my teeth checked",
    "Book a time for my general dental check",
    "I need an exam to check my teeth",
    "I want to make a dentist checkup",
    "I want to come in just to get my teeth checked",
    "Let me book a simple dental check",
    "I want to check if my teeth are healthy",
    "How can I book a routine visit?",
    "I’m due for an oral exam",
    "Book my dental check slot",
    "I want a basic cleaning and checkup",
    "Check my teeth, can I book for that?",
    "Please sign me up for a regular exam",
    "I’m interested in a dental review appointment",
    "I just want to make sure my teeth are okay"
],
    "ask_price" : [
        "How much does it cost for a dental cleaning?",
        "What do you charge for whitening?",
        "Is a checkup expensive?",
        "Can I get a price list?",
        "I want to know the price of a filling",
        "How much would a root canal be?",
        "Do you accept insurance?",
        "What’s the cost of a teeth cleaning?",
        "How much does a checkup usually cost?",
        "Tell me the price for a dental exam",
        "How much will whitening cost me?",
        "Can you tell me your rates?",
        "What’s the fee for a regular cleaning?",
        "What are your prices like?",
        "How expensive is it to whiten teeth?",
        "Are your services affordable?",
        "What does a dental visit cost?",
        "Do you have pricing for all treatments?",
        "Tell me the cost of dental work",
        "How much do you charge for appointments?",
        "Do you have a consultation fee?",
        "What are your prices for cleanings?",
        "How much is a full checkup?",
        "Are checkups covered by insurance?",
        "What’s the average cost of a visit?",
        "Can I see a price estimate?",
        "How much money will I need for whitening?",
        "What’s your rate for dental services?",
        "How much do your appointments run?",
        "Is there a cost for booking?",
        "What’s the price for kids’ cleanings?",
        "How much is a fluoride treatment?",
        "Can you break down your pricing?",
        "How much do you charge for dental X-rays?",
        "What’s the cost for routine cleaning?",
        "How expensive is a new patient exam?",
        "Do you charge extra for polishing?",
        "Is insurance required to reduce the cost?",
        "What’s your rate for tooth extractions?",
        "Can I pay with insurance for cleaning?",
        "What’s the pricing for braces?",
        "Are your prices fixed?",
        "What’s the price difference with and without insurance?",
        "Is there a student discount?",
        "How much do fillings usually cost?",
        "Is whitening part of the cleaning price?",
        "How much does an appointment usually cost?",
        "Can I get a quote for a checkup?",
        "How much does a consultation cost?",
        "Do you charge for no-shows?",
        "What’s the total cost of a cleaning and exam?",
        "Are your services billed per visit?",
        "Can I get a discount for paying cash?",
        "Do you offer payment plans?",
        "How much would it be without insurance?",
        "Are emergency visits more expensive?",
        "What’s your pricing policy?",
        "How much is the first visit?",
        "Can you tell me the cost breakdown?",
        "How much is a routine appointment?",
        "Are prices different on weekends?",
        "What do you charge for gum treatment?",
        "Is teeth whitening included in cleaning?",
        "What’s the rate for a full dental exam?",
        "How much do you charge per service?",
        "Is the price different for kids?",
        "How much would I pay for whitening?",
        "How much for just a cleaning, no exam?",
        "How much is it to remove plaque?",
        "What’s the cost for annual dental care?",
        "Are there extra fees I should know about?",
        "How do you price your appointments?",
        "What’s the cost of a full dental evaluation?",
        "How much does dental care cost here?",
        "Can you share your treatment costs?",
        "Do I need to pay upfront?",
        "Can you estimate the price of my visit?",
        "Do you offer free checkups?",
        "How much is teeth polishing?",
        "Do prices vary per doctor?",
        "Can you show me your pricing options?",
        "How much is a professional cleaning?",
        "Is whitening covered by insurance?",
        "Do prices change by season?",
        "What’s the current rate for cleanings?",
        "Is your price list online?",
        "How much should I expect to pay?",
        "How much do extra services cost?",
        "Do you offer free consultations?",
        "Is there a flat fee for exams?",
        "How much do I pay for a dental appointment?",
        "What’s the usual cost for whitening?",
        "Do you charge hourly?",
        "What is the price to clean teeth professionally?",
        "Can I see your cost per procedure?",
        "Do I pay more if I have no insurance?",
        "Are cleaning prices different for adults and kids?",
        "Do you offer bundled pricing?",
        "Can you give me a pricing estimate?",
        "How are your cleaning services priced?",
        "Do I pay more for a deep cleaning?",
        "How much do exams and cleanings cost together?",
        "How much does a basic dental visit cost?",
        "What’s your rate per visit?",
        "Do you provide low-cost options?",
        "Can you email me your pricing?",
        "What’s the total price if I get cleaning and x-rays?"
],
    "ask_availability": [
    "What appointment slots do you have?",
    "When’s your next available time?",
    "Do you have anything open this afternoon?",
    "What days are you available this week?",
    "When can I schedule something?",
    "Can I come in this weekend?",
    "Are you open on Saturday?",
    "Let me know what times are free",
    "Do you have anything early in the morning?",
    "Can I book something this evening?",
    "Any spots open today?",
    "Are you free on Monday?",
    "Is there an appointment time available right now?",
    "What’s your soonest available slot?",
    "When can I come in for a visit?",
    "Do you have any evening availability?",
    "Can I get an appointment before noon?",
    "Are mornings better for scheduling?",
    "What’s your schedule look like today?",
    "Can I get a spot this week?",
    "Are there any free appointments tomorrow?",
    "What are your available times?",
    "I’m looking for your open hours",
    "What days do you take appointments?",
    "Is there anything open next week?",
    "When’s the earliest I can come?",
    "Can I book something right away?",
    "What’s your calendar like?",
    "Can I come on Tuesday morning?",
    "Any chance you have a spot on Thursday?",
    "Do you take late appointments?",
    "When do you have room for a cleaning?",
    "I want to know your free times",
    "Is Friday morning available?",
    "When can I see the dentist?",
    "I need something midweek, what’s open?",
    "Do you have any open slots soon?",
    "What’s your next availability?",
    "Is anything available in the next few days?",
    "Can I book a spot this afternoon?",
    "Do you work on Sundays?",
    "What’s the best time you have open?",
    "Are appointments available this month?",
    "Do you have a free time next Thursday?",
    "I want to come in this week — when can I?",
    "What day can you fit me in?",
    "Are you free any time this weekend?",
    "Can I get in tomorrow morning?",
    "When’s the earliest available cleaning?",
    "What time slots are still open?",
    "Do you take appointments after 5 PM?",
    "Is your schedule open next Wednesday?",
    "Do you have walk-in times or open slots?"
],
    "tooth_pain": [
    "My tooth is aching a lot",
    "I'm getting tooth pain when I eat",
    "There's a sharp pain in my back molar",
    "My gums are swollen and sore",
    "I feel pressure in my jaw",
    "My tooth hurts when I drink cold water",
    "I'm having nonstop pain in one tooth",
    "I think I cracked a tooth",
    "It hurts when I bite down",
    "My teeth are super sensitive",
    "I'm experiencing throbbing tooth pain",
    "The left side of my mouth really hurts",
    "My jaw feels sore near my molars",
    "There's pain under one of my crowns",
    "I can’t sleep because of the toothache",
    "I feel a stinging pain in my tooth",
    "One of my fillings feels loose and painful",
    "My gum is inflamed and painful",
    "I think I have an abscess",
    "There’s a bump on my gum and it hurts",
    "My wisdom teeth are really hurting",
    "My mouth feels sore and tender",
    "Pain is radiating from my jaw to my ear",
    "My tooth hurts after eating sweets",
    "I think something's wrong with my root canal",
    "My face hurts on one side",
    "There’s pain when I touch my tooth",
    "My tooth is bleeding",
    "I think my tooth is infected",
    "It hurts to chew on one side",
    "There’s a bad taste coming from a sore tooth",
    "My tooth feels loose and painful",
    "Pain in my gum after flossing",
    "I feel heat sensitivity in my tooth",
    "The pain in my tooth is getting worse",
    "I feel throbbing under my crown",
    "I get a sharp jolt when I drink cold water",
    "My jaw pain is spreading to my temple",
    "I can't eat on one side because of pain",
    "One tooth is really bothering me",
    "I bit something and now my tooth hurts",
    "I feel pressure around one of my teeth",
    "My dental bridge is hurting my gum",
    "I need help with serious mouth pain",
    "My molars hurt",
    "I have pain after a dental procedure",
    "I feel soreness in my lower gum",
    "The pain keeps coming and going",
    "There’s pain when I touch the side of my face",
    "I feel discomfort after biting hard food",
    "My cheek is swollen from tooth pain",
    "I might have nerve pain in my tooth"
],

"book_filling": [
  "Can I book a filling appointment?",
  "I have a cavity that needs filling",
  "Please schedule me for a tooth filling",
  "I have to fill a cavity",
  "I have a couple of cavities that need to be filled",
  "I need a filling right now",
  "Please schedule me for a cavity filling",
  "My cavity hurts a lot",
  "My cavity is chipped and fell off, I need a replacement",
  "My cavity hurts a lot, I need it checked up",
  "The cavity needs to be filled with a black or white coating",
  "I need to book an apoointment to fill my tooth that has a cavity",
  "I want to fill a cavity" 
],

"book_extraction": [
    "I need to get a tooth removed",
    "Can you book me for wisdom tooth extraction?",
    "My wisdom teeth are hurting so much, I need them to get removed",
    "Need to remove my tooth please",
    "My teeth is rotting and needs to be removed",
    "I need to take out a couple of my teeth",
    "Do you guys do teeth removal",
    "My teeth needs to get replaced",
    "I'd like to schedule a tooth extraction",
    "Can I come in for a tooth removal?",
    "I want to get my molar pulled out",
    "One of my teeth needs to be extracted",
    "Can you help me book an appointment to remove a tooth?",
    "I have a decayed tooth that has to go",
    "I’m looking to remove my back teeth",
    "I need tooth extraction",
    "Please schedule me for a wisdom tooth removal",
    "I think my wisdom tooth is impacted and needs to come out",
    "I'm in pain and need to get a tooth pulled",
    "Can you help me take out my wisdom teeth?",
    "My dentist said I need a tooth extraction",
    "I have a bad tooth that has to be removed",
    "I want to get all four wisdom teeth removed",
    "Please set up an extraction appointment for me",
    "I can’t chew because of my bad tooth — need it taken out",
    "I’ve got a cracked tooth that must be extracted",
    "How do I book a tooth removal session?",
    "My gums are swollen and I need to get a tooth removed",
    "Can you remove infected teeth?",
    "I want to have a rotten tooth taken out",
    "My dentist recommended removing a tooth — can you help?",
    "I’m looking to schedule oral surgery for a tooth",
    "Do you remove teeth at your clinic?",
    "I need to take out a loose tooth",
    "I think my wisdom tooth is growing in sideways — I need it removed"
],

    "general_inquiry": [
    "What services are available at your clinic?",
    "Do you take new patients?",
    "Where are you located?",
    "Is your clinic open on Saturdays?",
    "How can I reach your office?",
    "Do you accept walk-in patients?",
    "Do you install braces?",
    "What kind of dental whitening do you offer?",
    "Is your clinic child-friendly?",
    "Can I have your contact number?",
    "Are consultations free?",
    "Do you have weekend hours?",
    "Can I visit without an appointment?",
    "Do you offer orthodontic care?",
    "What dental procedures do you do?",
    "What are your hours of operation?",
    "Do you treat toddlers?",
    "Can I call to ask questions?",
    "Do you provide cosmetic dentistry?",
    "What kind of insurance do you accept?",
    "Do you do extractions?",
    "Is your clinic accessible for disabled patients?",
    "Do you offer cleanings and exams?",
    "Can you tell me about your staff?",
    "Do you have multiple dentists?",
    "Do you take patients without insurance?",
    "Can I get a tour of your clinic?",
    "Do you treat dental emergencies?",
    "Are dental x-rays included in a checkup?",
    "What dental plans do you accept?",
    "Is your clinic busy usually?",
    "What types of treatments do you offer?",
    "Do you offer pediatric dentistry?",
    "Can I speak to a hygienist?",
    "Are you open on public holidays?",
    "Do you use modern equipment?",
    "Can I come in just to ask questions?",
    "What are your qualifications?",
    "Do you have parking available?",
    "Is your clinic near public transit?",
    "Do you offer sedation for anxious patients?",
    "What’s your cancellation policy?",
    "How many years have you been in practice?",
    "Are your services safe for pregnant women?",
    "Do you provide aftercare instructions?",
    "Is there a waiting list for appointments?",
    "Do you accept referrals?",
    "How long does a visit usually take?",
    "Are you open in the evenings?",
    "Do you have bilingual staff?"
],
   "out_of_scope": [
    "What’s the weather like tomorrow?",
    "Order me a pizza online",
    "How do I boil pasta?",
    "Tell me a funny joke",
    "My phone won’t charge",
    "What’s 45 divided by 5?",
    "Book me a flight to New York",
    "I need help with math homework",
    "What time is it in London?",
    "Is this a bakery?",
    "What’s the capital of Italy?",
    "How do I reset my laptop?",
    "Recommend a good restraunt",
    "Where can I buy running shoes?",
    "Turn off my smart lights",
    "Open YouTube for me",
    "Can you teach me to code?",
    "What’s the latest iPhone?",
    "Is soccer popular in Spain?",
    "How much does gas cost today?",
    "How do I unclog a sink?",
    "Can you set a timer?",
    "What is 7 x 9?",
    "Play some background music",
    "What's a good breakfast recipe?",
    "How do I find cheap flights?",
    "Show me today’s news headlines",
    "Can you check the stock market?",
    "Tell me a fun fact",
    "Translate ‘goodbye’ to Spanish",
    "How tall is the Eiffel Tower?",
    "How many states are in the USA?",
    "Remind me to take a walk later",
    "How do I install Zoom?",
    "What's trending on YouTube?",
    "Where is the closest train station?",
    "Can you play relaxing sounds?",
    "How do I bake cookies?",
    "Help me plan a vacation",
    "Is the mall open today?",
    "How many hours in a day?",
    "What’s a good TV show to binge?",
    "Can you give me directions?",
    "How fast can a cheetah run?",
    "What’s the deepest ocean?",
    "When was the Great Wall built?",
    "Can you open my camera app?",
    "What’s the tallest building in the world?",
    "Do I need an umbrella today?",
    "How do I update my software?"
]
}

# 2. Flatten the data into two lists
training_sentences = []
training_labels = []

for label, sentences in training_data.items():
    training_sentences.extend(sentences)
    training_labels.extend([label] * len(sentences))

# 3. Vectorize the text
vectorizer = CountVectorizer()
X_train = vectorizer.fit_transform(training_sentences)

# 4. Train the model
model = MultinomialNB()
model.fit(X_train, training_labels)

# Save model and vectorizer for session memory
with open("model.pkl", "wb") as f: pickle.dump(model, f)
with open("vectorizer.pkl", "wb") as f: pickle.dump(vectorizer, f)



# 2. Dialogue flows per intent (FSM)
dialogue_flows = {
    "book_cleaning": [
        {"prompt": "Perfect! First things first, what’s your full name?", "expect": "name"},
        {"prompt": "Sure! What day of the week would you like to come in?", "expect": "date"},
        {"prompt": "Got it. Do you prefer morning or afternoon?", "expect": "time_pref"},
        {"prompt": "Thanks! I've noted that down. Anything else you'd like to ask?", "expect": "end"}
    ],
    "book_filling": [
        {"prompt": "What’s your full name?", "expect": "name"},
        {"prompt": "We can take care of your cavity. What day works for you?", "expect": "date"},
        {"prompt": "Morning or afternoon?", "expect": "time_pref"},
        {"prompt": "Great! I've logged your appointment. Need anything else?", "expect": "end"}
    ],
    "book_extraction": [
        {"prompt": "What’s your full name?", "expect": "name"},
        {"prompt": "Ouch! When would you like to come in for a tooth extraction?", "expect": "date"},
        {"prompt": "Do you prefer a morning or afternoon extraction?", "expect": "time_pref"},
        {"prompt": "Noted! We'll get that taken care of. Anything else you’d like to do?", "expect": "end"}
    ],
    "book_checkup": [
        {"prompt": "What’s your full name?", "expect": "name"},
        {"prompt": "Let’s keep those teeth healthy! When would you like your checkup?", "expect": "date"},
        {"prompt": "Morning or afternoon for your checkup?", "expect": "time_pref"},
        {"prompt": "Checkup scheduled! Anything else I can help with?", "expect": "end"}
    ],
    "book_whitening": [
        {"prompt": "What’s your full name?", "expect": "name"},
        {"prompt": "Brighten your smile! What day works for whitening?", "expect": "date"},
        {"prompt": "Would you like it in the morning or afternoon?", "expect": "time_pref"},
        {"prompt": "Got it! Whitening is scheduled. Need anything else?", "expect": "end"}
    ],
    "book_root_canal": [
        {"prompt": "What’s your full name?", "expect": "name"},
        {"prompt": "Root canal needed — let’s book you in. What day is good?", "expect": "date"},
        {"prompt": "Morning or afternoon for your root canal?", "expect": "time_pref"},
        {"prompt": "All set! Let us know if you need anything else.", "expect": "end"}
    ],
    "book_braces_consult": [
        {"prompt": "What’s your full name?", "expect": "name"},
        {"prompt": "Let’s get you a braces consultation. When are you available?", "expect": "date"},
        {"prompt": "Morning or afternoon for the consult?", "expect": "time_pref"},
        {"prompt": "Great! We’ll discuss your options then. Anything else?", "expect": "end"}
    ],
    "cancel_appointment": [
        {"prompt": "What’s your full name?", "expect": "name"},
        {"prompt": "No problem. What appointment would you like to cancel (cleaning, checkup, etc.)?", "expect": "type"},
        {"prompt": "Got it. Anything else you’d like to do?", "expect": "end"}
    ],
    "reschedule_appointment": [
        {"prompt": "What’s your full name?", "expect": "name"},
        {"prompt": "Sure. What type of appointment are you rescheduling?", "expect": "type"},
        {"prompt": "What’s your new preferred date?", "expect": "date"},
        {"prompt": "Morning or afternoon for the new time?", "expect": "time_pref"},
        {"prompt": "Rescheduled! Let me know if you need anything else.", "expect": "end"}
    ],
    "tooth_pain": [
        {"prompt": "What’s your full name?", "expect": "name"},
        {"prompt": "I’m sorry to hear that. Would you like to book an emergency visit?", "expect": "yes_no"},
        {"prompt": "When would you like to come in?", "expect": "date"},
        {"prompt": "Morning or afternoon?", "expect": "time_pref"},
        {"prompt": "We’ll see you soon. Take care until then!", "expect": "end"}
    ],
    "ask_price": [
        {"prompt": "Sure. What treatment are you asking about (cleaning, whitening, extraction, or is it something else)?", "expect": "type"},
        {"prompt": "Let me look that up for you.", "expect": "end"}
    ],
    "ask_availability": [
        {"prompt": "Let me check our calendar. What type of service are you interested in?", "expect": "type"},
        {"prompt": "Do you prefer a morning or afternoon appointment?", "expect": "time_pref"},
        {"prompt": "Thanks! We’ll get back to you with availability.", "expect": "end"}
    ],
    "general_inquiry": [
        {"prompt": "Sure, I can help with info about our services. What would you like to know?", "expect": "topic"},
        {"prompt": "Thanks for reaching out!", "expect": "end"}
    ],
    "out_of_scope": [
        {"prompt": "Sorry, I can only help with dental-related questions. Try asking about appointments or treatments.", "expect": "end"}
    ]
}





fillers = {
  "book_cleaning": [
    "When are you available?",
    "Do you have a day in mind?",
    "Let me know your preferred time."
  ],
  "ask_price": [
    "Which treatment are you curious about?",
    "Are you asking about cleaning, whitening, or something else?",
    "Let me know what you're looking for pricing on."
  ],
  "cancel_appointment": [
    "Let us know if you'd like to reschedule.",
    "We hope to see you another time!",
    "You're welcome to rebook any time."
  ],
  "clarify_intent": [
    "Maybe try rephrasing?",
    "Could you say that a little differently?",
    "I'm here to help with dental services and appointments!"
  ],
  "book_whitening": [
    "Do you have a time in mind for your whitening?",
    "We can brighten your smile — when works for you?",
    "Let me know your availability so we can whiten those teeth!"
  ],
  "book_checkup": [
    "Do you prefer mornings or afternoons?",
    "Let's keep those teeth healthy — when can you come in?",
    "Tell me the best time for your checkup."
  ],
  "reschedule_appointment": [
    "Let me know the new time that works for you.",
    "We can easily move your appointment.",
    "What day or time would you like instead?"
  ],
  "ask_availability": [
    "Are you looking for something this week?",
    "Any specific day or time you're hoping for?",
    "We’ll check our calendar — what works best for you?"
  ],
  "tooth_pain": [
    "That sounds painful. Would you like to book an urgent visit?",
    "Let’s get that looked at. Should I find you a time today?",
    "Would you like help booking an emergency dental appointment?"
  ],
  "general_inquiry": [
    "Feel free to ask anything about our clinic.",
    "I’m here to help with any questions you have.",
    "Ask away — I can help with services, hours, or anything else."
  ],
  "book_filling": [
    "We can get that cavity taken care of — when are you free?",
    "Let’s book a time to fix your filling.",
    "Do you want to come in this week for the filling?"
  ],
  "book_extraction": [
    "Let’s find a time to remove that tooth.",
    "Would you like an extraction appointment this week?",
    "We can help — what day works for you?"
  ],
  "book_root_canal": [
    "Let’s book you in before it gets worse.",
    "We’ll take care of that root canal. What day do you prefer?",
    "Would you like to come in this week or next?"
  ],
  "book_braces_consult": [
    "We can help you explore braces or Invisalign options.",
    "When would you like to come in for your orthodontic consultation?",
    "Let’s schedule a time to talk about braces."
  ],
  "out_of_scope": [
    "Let’s stick to dental topics!",
    "I’m built to help with dental care — try asking about appointments or pricing.",
    "Please ask me something related to dentistry."
  ]
}

conversation_context = {
    "last_intent": None,
    "step": 0,
    "params": {},
    "history": []
}


TREATMENT_DURATIONS = {
    "teeth_whitening": 90,  # 1.5 hours
    "book_whitening": 90,
    "book_cleaning": 60,
    "book_checkup": 60,
    "book_filling": 60,
    "book_extraction": 60,
    "book_root_canal": 60,
    "book_braces_consult": 60,
}


# 6. Predict the intent    
def predict_intent(user_input):
    with open("model.pkl", "rb") as f: model = pickle.load(f)
    with open("vectorizer.pkl", "rb") as f: vectorizer = pickle.load(f)
    X_test = vectorizer.transform([user_input])
    prediction = model.predict(X_test)[0]
    confidence = model.predict_proba(X_test).max()
    return prediction, confidence


# Slot Filling (very basic extraction)


def extract_slot(user_input):
    slots = {}

    # --- Clean input before parsing ---
    # e.g., "Friday's good" -> "Friday good"
    cleaned_input = re.sub(r"\b([A-Za-z]+)'s\b", r"\1", user_input)  # possessive fix
    cleaned_input = re.sub(r"[^A-Za-z0-9\s:]", "", cleaned_input)    # remove punctuation

    # --- Date & Time Parsing ---
    parsed_date = dateparser.parse(cleaned_input, settings={"PREFER_DATES_FROM": "future"})
    if parsed_date:
        day_name = parsed_date.strftime('%A').lower()
        slots["date"] = day_name

        hour = parsed_date.hour
        if 5 <= hour < 12:
            slots["time_pref"] = "morning"
        elif 12 <= hour < 17:
            slots["time_pref"] = "afternoon"
        else:
            slots["time_pref"] = "evening"

    # --- Time preference keywords ---
    lowered = cleaned_input.lower()
    if "morning" in lowered:
        slots["time_pref"] = "morning"
    elif "afternoon" in lowered:
        slots["time_pref"] = "afternoon"
    elif "evening" in lowered:
        slots["time_pref"] = "evening"

    # --- Name Detection (Safe) ---
    # Explicit phrases like "my name is Ali Khan"
    name_match = re.search(
        r"\b(?:my name is|i am|i'm|this is)\s+([A-Z][a-z]+(?: [A-Z][a-z]+)?)",
        user_input,
        re.IGNORECASE
    )
    if name_match:
        name_candidate = name_match.group(1).strip()
        # Avoid accidentally capturing common treatment keywords
        if name_candidate.lower() not in {
            "cleaning", "whitening", "checkup", "extraction", "filling",
            "root canal", "consultation"
        }:
            slots["name"] = name_candidate

    else:
        # Fallback for direct input like "Ali Khan"
        words = user_input.strip().split()
        if len(words) == 2 and all(w[0].isupper() for w in words if w.isalpha()):
            name_candidate = user_input.strip()
            if name_candidate.lower() not in {
                "cleaning", "whitening", "checkup", "extraction", "filling",
                "root canal", "consultation"
            }:
                slots["name"] = name_candidate

    return slots






def get_dynamic_response(intent):
    response_templates = [
        "Sure! Let's take care of that. {filler}",
        "Absolutely. {filler}",
        "Happy to help! {filler},"
    ]
    template = choice(response_templates)
    filler = choice(fillers.get(intent, [""]))
    return template.format(filler=filler)

# 6. FSM-driven response engine
import requests

import requests


def handle_response(user_input):
    context = conversation_context
    context["history"].append(user_input)

    # --- Step 1: Predict intent and extract new slots ---
    new_prediction, confidence = predict_intent(user_input)
    found_slots = extract_slot(user_input)

    # --- Step 2: FSM setup ---
    current_intent = context.get("last_intent")
    step = context.get("step", 0)
    flow = dialogue_flows.get(current_intent, [])
    expected_slot = flow[step]["expect"] if step < len(flow) else None
    slot_was_expected = expected_slot in found_slots

    # --- Step 3: Disable mode switching completely ---
    # Instead of switching, we always continue with the currently active intent.
    # This avoids bugs when the user types "My name is..." or "yes" and the system mistakenly switches.
    new_prediction = current_intent

    # --- Step 4: First time setup ---
    if step == 0:
        prediction = new_prediction or predict_intent(user_input)[0]
        context["last_intent"] = prediction
        context["params"] = found_slots

        flow = dialogue_flows.get(prediction, [])
        required_slots = [s["expect"] for s in flow if s["expect"] != "end"]
        missing = [s for s in required_slots if s not in context["params"]]

        if not flow:
            return f"Okay, you want to {prediction.replace('_', ' ')}. Let me help with that."

        # Set FSM to first missing slot step
        if missing:
            for i, step_info in enumerate(flow):
                if step_info["expect"] == missing[0]:
                    context["step"] = i
                    break
        else:
            context["step"] = len(flow)

    else:
        prediction = current_intent
        context["params"].update(found_slots)

    print(f"Current intent: {context['last_intent']}, Step: {context['step']}, Slots: {context['params']}")

    # --- Step 5: Check if all required slots are filled ---
    flow = dialogue_flows.get(prediction)
    step = context["step"]

    required_slots = [s["expect"] for s in flow if s["expect"] != "end"]
    missing = [s for s in required_slots if s not in context["params"]]

    if not missing:
        context["step"] = len(flow)

    # --- Step 6: Booking completed, submit to backend ---
    if context["step"] >= len(flow):
        name = context['params'].get('name', 'John Doe')
        date = context['params']['date'].capitalize()
        time_pref = context['params']['time_pref']

        default_times = {
            "morning": "09:00 AM",
            "afternoon": "01:00 PM",
            "evening": "04:00 PM"
        }
        chosen_time = default_times.get(time_pref, "09:00 AM")

        treatment = prediction.replace("book_", "").replace("_", " ").title()
        duration = TREATMENT_DURATIONS.get(prediction, 60)

        payload = {
            "name": name,
            "date": date,
            "time": chosen_time,
            "treatment": treatment,
            "duration": duration
        }

        try:
            response = requests.post("http://127.0.0.1:8000/api/add_appointment", json=payload)
            if response.status_code == 200:
                confirmation = f"You are booked for {treatment} on {date} at {chosen_time}."
            else:
                confirmation = "There was an error while trying to book your appointment."
        except Exception:
            confirmation = "Could not connect to the booking server."

        parsed_info = (
            f"[Parsed Info] Name: {name}, Date: {date}, Time: {chosen_time}, "
            f"Treatment: {treatment}, Duration: {duration} mins"
        )

        # Reset FSM context after booking
        context["step"] = 0
        context["params"] = {}
        context["last_intent"] = None

        return confirmation + "\n" + parsed_info + "\nIs there anything else I can help you with?"

    # --- Step 7: Continue FSM (next prompt) ---
    expected_slot = flow[step]["expect"]
    if expected_slot in context["params"]:
        context["step"] += 1
        return handle_response(user_input)
    else:
        return flow[step]["prompt"]



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

