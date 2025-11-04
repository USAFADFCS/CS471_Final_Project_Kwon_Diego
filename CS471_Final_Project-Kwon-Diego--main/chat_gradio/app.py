import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer
from threading import Thread
from Schedule.Sleep_Tracker import SleepManager

def organize_activities_by_day(activity_text):
    import json
    try:
        activities = json.loads(activity_text)
    except json.JSONDecodeError:
        return {"error": "Invalid input. Please provide a JSON list of activities."}

    schedule = {}
    for act in activities:
        day = act.get("day", "unspecified")
        task = act.get("task", "No task provided")
        schedule.setdefault(day, []).append(task)

    
    sleep_manager = SleepManager()
    
    sleep_manager.sleepmanager.addSleep(8)

    if not sleep_manager.correctSleepHours():
        for day in schedule:
            schedule[day].append("Sleep (8 hours)")

    return schedule

import sys
import os

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
from schedule.Sleep_Tracker import SleepManager

from datetime import datetime, timedelta

from datetime import datetime, timedelta

def parse_time_range(time_range):
    """Convert 'HH:MM-HH:MM' to start and end datetime objects."""
    start_str, end_str = time_range.split('-')
    start = datetime.strptime(start_str, "%H:%M")
    end = datetime.strptime(end_str, "%H:%M")
    if end <= start:  # handle sleep past midnight
        end += timedelta(days=1)
    return start, end

def organize_activities_by_day(schedule_list):
    """
    Takes a list of tuples (Day, Event, 'Start-End') and ensures 8 hours of sleep per day.
    Example: [("Monday", "Work", "09:00-17:00"), ("Monday", "Sleep", "23:00-07:00")]
    """
    schedule = {}
    sleep_manager = SleepManager()  # from your imported module

    for day, event, time_range in schedule_list:
        start, end = parse_time_range(time_range)
        duration = (end - start).total_seconds() / 3600  # convert seconds to hours
        schedule.setdefault(day, []).append(f"{event}: {time_range}")

        if event.lower() == "sleep":
            sleep_manager.sleepmanager.addSleep(duration)

    results = {}
    for day, events in schedule.items():
        total_sleep = sleep_manager.sleepmanager.getSleep()
        if total_sleep >= 8:
            results[day] = events + [f"Adequate sleep: {total_sleep:.2f} hours"]
        else:
            results[day] = events + [f"⚠ Inadequate sleep: {total_sleep:.2f} hours (add more sleep)"]

    return results

# Loading the tokenizer and model from Hugging Face's model hub.
tokenizer = AutoTokenizer.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")
model = AutoModelForCausalLM.from_pretrained("TinyLlama/TinyLlama-1.1B-Chat-v1.0")

# using CUDA for an optimal experience
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


# Defining a custom stopping criteria class for the model's text generation.
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [2]  # IDs of tokens where the generation should stop.
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:  # Checking if the last generated token is a stop token.
                return True
        return False


# Function to generate model predictions.
def predict(message, history):
    history_transformer_format = history + [[message, ""]]
    stop = StopOnTokens()

    # Formatting the input for the model.
    messages = "</s>".join(["</s>".join(["\n<|user|>:" + item[0], "\n<|assistant|>:" + item[1]])
                        for item in history_transformer_format])
    model_inputs = tokenizer([messages], return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=1024,
        do_sample=True,
        top_p=0.95,
        top_k=50,
        temperature=0.7,
        num_beams=1,
        stopping_criteria=StoppingCriteriaList([stop])
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()  # Starting the generation in a separate thread.
    partial_message = ""
    for new_token in streamer:
        partial_message += new_token
        if '</s>' in partial_message:  # Breaking the loop if the stop token is generated.
            break
        yield partial_message


# Setting up the Gradio chat interface.
gr.ChatInterface(predict,
                 title="Tinyllama_chatBot",
                 description="Ask Tiny llama any questions",
                 examples=['How to cook a fish?', 'Who is the president of US now?']
                 ).launch()  # Launching the web interface.