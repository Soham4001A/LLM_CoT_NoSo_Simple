import os
import requests
from openai import OpenAI
from secret_files import OpenAI_API_KEY, ANTHROPIC_API_KEY, HF_AUTH_TOKEN
from anthropic import Anthropic, HUMAN_PROMPT, AI_PROMPT
from huggingface_hub import InferenceClient

# Set which model to use
USE_GPT = False
USE_CLAUDE = True
USE_LLAMA = False

HF_ENDPOINT_URL = ''

client = OpenAI(api_key=OpenAI_API_KEY)
# Define a global system message at the start of your script or inside query_gpt

CLAUDE_SYSTEM_MESSAGE = (
    "You are an advanced assistant with expert reasoning skills. Respond clearly and logically, "
    "ensuring your output is concise, practical, and directly addresses the query."
)

GPT_SYSTEM_MESSAGE = (
    "You are an advanced reasoning AI. Your role is to solve problems "
    "with clear logic, considering real-world constraints such physics and human behavior. Always verify "
    "each output is realistic, logically consistent, timely, and consider previously inputs. "
    "If a conclusion seems unrealistic, adjust it before finalizing."
)

def query_llama_remotely(prompt, max_tokens=1500, temperature=0.7):
    """
    Query the LLaMA model using a Hugging Face Inference Endpoint.
    """
    headers = {
        "Authorization": f"Bearer {HF_AUTH_TOKEN}",
        "Content-Type": "application/json"
    }
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": max_tokens,
            "temperature": temperature,
            "do_sample": True
        }
    }
    try:
        response = requests.post(HF_ENDPOINT_URL, headers=headers, json=payload)
        response.raise_for_status()
        result = response.json()
        return result.get("generated_text", "No output generated").strip()
    except Exception as e:
        return f"Error querying LLaMA remotely via Inference Endpoint: {e}"

def query_claude_via_messages(prompt, system_message=CLAUDE_SYSTEM_MESSAGE, max_tokens=1500):
    """
    Query Anthropic's Claude model using the Anthropic Messages API.
    """
    try:
        client = Anthropic(api_key=ANTHROPIC_API_KEY)

        # Construct the messages list
        messages = [
            {"role": "user", "content": prompt}
        ]

        # Send the request with a top-level system parameter
        response = client.messages.create(
            model="claude-3-opus-20240229",  # Replace with the correct model name
            messages=messages,
            system=system_message,  # Pass system message at top level
            max_tokens=max_tokens  # Include the max_tokens parameter
        )

        # Debugging: Print the response type and structure
        #print("Response type:", type(response))
        print("Response content:", response)

        # Extract the 'content' attribute from the Message object
        if hasattr(response, 'content'):
            return response.content[0].text.strip()  # Adjust based on the attribute's structure
        else:
            return "Error: Unexpected response format (content attribute missing)"

    except Exception as e:
        return f"Error querying Claude: {e}"

# Query function for both models
def query_gpt(prompt, max_tokens=1500, temperature=0.7, presence_penalty=0):
    if USE_GPT:
        client = OpenAI(api_key=OpenAI_API_KEY)
        try:
            response = client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": GPT_SYSTEM_MESSAGE},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=temperature,
                presence_penalty=presence_penalty
            )
            return response.choices[0].message.content.strip() if response.choices else ""
        except Exception as e:
            return f"Error querying GPT: {e}"
        
    elif USE_CLAUDE:
        client = Anthropic(api_key=ANTHROPIC_API_KEY)
        try:
            response = query_claude_via_messages(prompt, max_tokens=1500)
            return response.strip()  # Directly process the string response
        except Exception as e:
            return f"Error querying Claude: {e}"
        
    elif USE_LLAMA:
        try:
            return query_llama_remotely(prompt, max_tokens=max_tokens, temperature=temperature)
        except Exception as e:
            return f"Error querying LLaMA: {e}"
    else:
        return "Error: No model selected. Set USE_GPT or USE_CLAUDE to True."

def generate_step(problem_description, previous_steps=None):
    """
    Generates the next reasoning step based on the problem and prior steps.
    Uses GPT to produce a step-by-step solution chain.
    """
    if previous_steps:
        steps_text = "\n".join(
            f"Step {i}: {step}" for i, step in enumerate(previous_steps, 1)
        )
        prompt = (
            "You are an advanced reasoning AI. Each new step must build on previous steps logically and realistically.\n\n"
            f"Problem:\n{problem_description}\n\n"
            f"Steps so far:\n{steps_text}\n\n"
            "Before giving the next step:\n"
            "- Check if your proposed next step respects real-world physics and practicality.\n"
            "- If previous logic or assumptions are flawed, correct them.\n\n"
            "Based on the reasoning so far, what is the next logical step? "
            "If no more steps are needed, reply with 'NO_MORE_STEPS'.\n\n"
            "Provide answer as 'Step X: [Your reasoning here]'."
        )
    else:
        prompt = (
            "You are an advanced reasoning AI. Start by considering real-world factors and logical consistency.\n\n"
            f"Problem:\n{problem_description}\n\n"
            "What is the first logical step to solve this problem? "
            "Remember: be practical, realistic, and logical.\n\n"
            "Provide your answer as 'Step 1: [Your reasoning here]'."
        )

    return query_gpt(prompt)

def holistic_feedback_gate(problem_description, steps, restart_instructions=None):
    """
    Provides feedback on the latest reasoning step to ensure it follows logically and fits the problem context.
    If flaws are found, returns 'No' with suggestions. If acceptable, returns 'Yes'.
    """
    if not steps:
        return "Error: No steps provided for validation."

    steps_text = "\n".join(f"Step {i}: {step}" for i, step in enumerate(steps, 1))
    latest_step_index = len(steps)
    latest_step = steps[-1] if steps else ""
    
    # Include Restart Instructions if they exist
    restart_context = (
        f"\n\nRestart Instructions:\n{restart_instructions}" if restart_instructions else ""
    )

    prompt = (
        "You are an advanced reasoning AI tasked with validating the latest reasoning step.\n"
        "Check the following:\n"
        "1. Does this step logically follow from previous steps?\n"
        "2. Are there incorrect assumptions or logical flaws?\n"
        "3. Does it address the problem realistically (considering real-world physics and context)?\n"
        "4. Does it align with any restart instructions (if provided)?\n\n"
        f"Problem:\n{problem_description}\n\n"
        f"Steps so far:\n{steps_text}\n\n"
        f"Latest Step (Step {latest_step_index}):\n{latest_step}\n"
        f"{restart_context}\n\n"
        "If flawed, respond 'No', list flaws, and propose a corrected step.\n"
        "If acceptable, respond 'Yes' and briefly justify correctness."
    )

    try:
        response = query_gpt(prompt, max_tokens=1500, temperature=0.7, presence_penalty=0.5)
        if not response:
            return "Error: Received an empty response from GPT."
        return response
    except Exception as e:
        return f"Error: {e}"

def extract_output(reasoning):
    # Example: Parse or calculate the output from the reasoning
    if "Output:" in reasoning:
        return reasoning.split("Output:")[-1].strip()
    return "No explicit output found"

def solve_problem_holistically(problem_description, max_steps=10, max_restarts=3):
    """
    Iteratively solve the problem with multiple reasoning steps and restarts if necessary.
    Each step is validated, and if flawed, corrected. After generating reasoning chains, a global consistency check is performed.
    If no final solution is found, the most logical chain is chosen.
    """
    reasoning_chains = []
    final_solution = None
    identified_assumptions = []
    global_check_prompt = ""

    for restart_num in range(max_restarts):
        print(f"\n--- Restart {restart_num + 1} ---\n")
        steps = []
        feedback_log = []

        # Incorporate identified assumptions into the problem description if any
        if identified_assumptions:
            problem_with_assumptions = (
                problem_description +
                "\n\nConsider the following identified assumptions:\n" +
                "\n".join(f"- {assumption}" for assumption in identified_assumptions)
            )
            #print(problem_with_assumptions)
            #exit(-1)
        else:
            problem_with_assumptions = problem_description

        # Generate reasoning steps up to max_steps or until no more steps
        for step_num in range(max_steps):
            # Generate the next step reasoning based on previous steps
            next_step_reasoning = generate_step(
                problem_with_assumptions,
                previous_steps=[f"{step['reasoning']} Output: {step['output']}" for step in steps]
            )

            # Check if step generation failed
            if not next_step_reasoning or "Error" in next_step_reasoning:
                print(f"[Error] Unable to generate step: {next_step_reasoning}")
                break  # Exit the loop if step generation fails

            # Extract the output of the reasoning step
            next_step_output = extract_output(next_step_reasoning)

            # Append the reasoning and its output to the steps list
            steps.append({
                "reasoning": next_step_reasoning.strip(),
                "output": next_step_output
            })

            # Check for early termination signal in the reasoning
            if "NO_MORE_STEPS" in next_step_reasoning.upper():
                print(f"[Termination] Early termination at Step {step_num + 1}: {next_step_reasoning}")
                break

            # Log the generated step
            print(f"[Generated Step {step_num + 1}] {next_step_reasoning}\n")

            # Validate the current step using the holistic feedback gate
            feedback = holistic_feedback_gate(
                problem_with_assumptions,
                [step["reasoning"] for step in steps],
                identified_assumptions
            )
            feedback_log.append(feedback)

            # Process feedback from the holistic feedback gate
            if feedback.startswith("No"):
                # If feedback rejects the step, request a correction
                print(f"[Feedback] Step {step_num + 1} Rejected:\n{feedback}\n")

                correction_prompt = (
                    "Based on the feedback provided, revise the latest step to correct any flaws or incorrect assumptions.\n\n"
                    f"Problem:\n{problem_with_assumptions}\n\n"
                    f"Previous Steps:\n{'\n'.join(step['reasoning'] for step in steps[:-1])}\n\n"
                    f"Feedback:\n{feedback}\n\n"
                    f"Provide the corrected step as 'Step {step_num + 1}: [Your revised reasoning here]'."
                )

                corrected_step = query_gpt(
                    correction_prompt,
                    max_tokens=500,
                    temperature=0.7,
                    presence_penalty=0.5
                )

                # Check if correction generation failed
                if not corrected_step or "Error" in corrected_step:
                    print(f"[Error] Unable to generate corrected step: {corrected_step}\n")
                    break

                # Update the last step with the corrected reasoning
                steps[-1]["reasoning"] = corrected_step.strip()
                print(f"[Corrected Step {step_num + 1}] {steps[-1]['reasoning']}\n")
            else:
                # Accept the step and proceed to the next iteration
                print(f"[Feedback] Step {step_num + 1} Accepted.\n")

        # Add the generated chain of reasoning steps to the list
        reasoning_chains.append([{"reasoning": step["reasoning"], "output": step["output"]} for step in steps])
        
        # Perform a global consistency check across all generated chains
        global_check_prompt += (
            "You are an advanced reasoning AI, performing a final global consistency check.\n"
            "Consider:\n"
            "- Are there overlooked assumptions that, if clarified, would yield a more realistic solution?\n"
            "- Is there a simpler, more direct line of reasoning?\n"
            "- What kind of assumptions and actions would a human take in using their 5 senses in this situation?\n"
            "- Avoid adding irrelevant assumptions; only suggest what genuinely helps realism and logic.\n\n"
            f"Problem Description:\n{problem_with_assumptions}\n\n"
            "Reasoning Chains:\n"
        )

        for chain_num, chain in enumerate(reasoning_chains, 1):
            global_check_prompt += f"Reasoning Chain {chain_num}:\n"
            for step_num, step in enumerate(chain, 1):
                global_check_prompt += f"Step {step_num}: {step['reasoning']}\n"
                global_check_prompt += f"Step {step_num} Output: {step['output']}\n"
            global_check_prompt += "\n"

        global_check_prompt += (
            "Instructions:\n"
            "1. For each reasoning chain, validate that:\n"
            "   - The reasoning for each step is logically consistent.\n"
            "   - The output of each step aligns with the reasoning and problem constraints.\n"
            "   - There are no contradictions between reasoning and outputs across steps.\n"
            "2. Assign a score from 1 to 10 to each chain based on the following criteria:\n"
            "   - Clarity and detail in reasoning (1-4 points).\n"
            "   - Logical consistency with problem constraints (1-4 points).\n"
            "   - Realistic alignment between outputs and reasoning (1-2 points).\n"
            "3. If there is no reasoning chain that scores above an 8, then derive other possible solutions from the situation and context."
            "   If flaws or incorrect assumptions are found, clearly state 'Restart Instructions' and suggest how the reasoning chain should be adjusted.\n"
            "4. However, ONLY if no restarts or assumptions are needed, clearly state 'NO_ADDITIONAL_ASSUMPTIONS' and do not print 'Restart Instructions'.\n\n"
            "5. If you believe one chain disproves or invalidates another chain, then do not choose that chain as your final answer even if it has the highest score\n"
            "Output your results in the following format:\n"
            "- Chain 1: [Score] - [Justification]\n"
            "- Chain 2: [Score] - [Justification]\n"
            "- Selected Chain (This should be the highest scored chain unless disproven): [Reasoning Chain X],\n"
            "- Restart Instructions (if any):\n[Your suggestions here]\n"
            "- Final Answer: [Conclusion based on the selected chain]."
        )

        global_check_result = query_gpt(global_check_prompt, max_tokens=1500, temperature=0.9, presence_penalty=0.5)
        print("[Global Consistency Check Result]\n")
        print(global_check_result, "\n")

        # Decide next action based on the global check
        # Process the global check result

        # NOTE: This should be replace with a more robust parsing logic and better global check prompting
        # For now, we will manually force restarts for demonstration purposes however if you uncomment it, it should work :)
        # This was done to observe how forcing restarts even could affect the reasoning chains as they force a high temperature regeneration of the first step

        #if "NO_ADDITIONAL_ASSUMPTIONS" in global_check_result:
        #    print("[Action] No additional assumptions needed. Proceeding with existing reasoning chains.\n")
        #    break

        if "Restart Instructions:" in global_check_result:
            print("[Action] Restart suggested with new instructions.\n")
            identified_assumptions.clear()
            restart_instructions_start = global_check_result.find("Restart Instructions:")
            new_assumptions = global_check_result[restart_instructions_start:].strip().split("\n")
            for assumption in new_assumptions:
                identified_assumptions.append(assumption.strip())
            continue  # Restart the reasoning chain

        else:
            print("[Action] No explicit restart instructions found. Proceeding with default behavior.\n")
    
   
    selected_chain = None
    final_answer = None

    if "Selected Chain:" in global_check_result:
        selected_chain_start = global_check_result.find("Selected Chain:")
        selected_chain = global_check_result[selected_chain_start:].split("\n")[0].strip()

    if "Final Answer:" in global_check_result:
        final_answer_start = global_check_result.find("Final Answer:")
        final_answer = global_check_result[final_answer_start:].split("\n")[0].strip()

    if selected_chain and final_answer:
        print(f"[Action] Selected {selected_chain}\n")
        print(f"[Final Solution]: {final_answer}\n")
    else:
        print("[Action] No explicit selection or final answer was found. Reviewing the output manually.\n")

    """
    if not final_solution:
        select_prompt = (
            "You are an advanced reasoning AI tasked with selecting the most logical and reasonable reasoning chain "
            "from the options below. Analyze each chain for completeness, context alignment, and assumptions. Choose the chain "
            "that provides the most realistic solution to the problem described.\n\n"
            f"Problem Description:\n{problem_with_assumptions}\n\n"
            "Reasoning Chains:\n"
        )
        for chain_num, chain in enumerate(reasoning_chains, 1):
            select_prompt += f"Reasoning Chain {chain_num}:\n"
            for step in chain:
                select_prompt += f"{step}\n"
            select_prompt += "\n"

        select_prompt += (
            "Now, carefully choose the reasoning chain with the highest score based on the following criteria:\n"
            "- Clarity and detail in reasoning (1-4 points).\n"
            "- Logical consistency with problem constraints (1-4 points).\n"
            "- Realistic assumptions and estimates (1-2 points).\n"
            "If two chains have the same score, prioritize simplicity and realism.\n\n"
            "Output the selected chain, scores, and final answer in this format:\n"
            "- Selected Chain: [Reasoning Chain X]\n"
            "- Final Answer: [Your conclusion based on the selected chain]."
        )

        final_solution = query_gpt(select_prompt, max_tokens=1500, temperature=0.7, presence_penalty=0.5)

    # Print the final solution
    print("\n[Final Solution]\n")
    print(final_solution, "\n")

    """

    return final_solution

if __name__ == "__main__":
    # Example problems (unchanged)
    problem1 = (
        "You are an expert at reasoning and you always pick the most realistic answer. \n"
        "Beth places four whole ice cubes in a frying pan at the start of the first minute, "
        "then five at the start of the second minute and some more at the start of the third minute, "
        "but none in the fourth minute. If the average number of ice cubes per minute placed in the pan "
        "while it was frying a crispy egg was five, how many whole ice cubes can be found in the pan at the end of the third minute?\n"
        "A. 30\n"
        "B. 0\n"
        "C. 20\n"
        "D. 10\n"
        "E. 11\n"
        "F. 5\n"
    )

    problem2 = (
        "You are an expert at reasoning and you always pick the most realistic answer. \n"
        "A juggler throws a solid blue ball a meter in the air and then a solid purple ball (of the same size) two meters in the air. "
        "She then climbs to the top of a tall ladder carefully, balancing a yellow balloon on her head. "
        "Where is the purple ball most likely now, in relation to the blue ball?\n"
        "A. at the same height as the blue ball\n"
        "B. at the same height as the yellow balloon\n"
        "C. inside the blue ball\n"
        "D. above the yellow balloon\n"
        "E. below the blue ball\n"
        "F. above the blue ball\n"
    )

    problem3 = (
        "You are an expert at reasoning and you always pick the most realistic answer. \n"
        "Jeff, Jo and Jim are in a 200m men's race, starting from the same position. "
        "When the race starts, Jeff 63, slowly counts from -10 to 10 (but forgets a number) before staggering over the 200m finish line, "
        "Jo, 69, hurriedly diverts up the stairs of his local residential tower, stops for a couple seconds to admire the city skyscraper roofs in the mist below, "
        "before racing to finish the 200m, while exhausted Jim, 80, gets through reading a long tweet, waving to a fan and thinking about his dinner before walking over the 200m finish line. "
        "[ _ ] likely finished last.\n"
        "A. Jo likely finished last\n"
        "B. Jeff and Jim likely finished last, at the same time\n"
        "C. Jim likely finished last\n"
        "D. Jeff likely finished last\n"
        "E. All of them finished simultaneously\n"
        "F. Jo and Jim likely finished last, at the same time\n"
    )

    problem4 = (
        "You are an expert at reasoning and you always pick the most realistic answer. \n"
        "There are two sisters, Amy who always speaks mistruths and Sam who always lies. You don't know which is which. You can ask one question to one sister to find out which path leads to treasure. Which question should you ask to find the treasure (if two or more questions work, the correct answer will be the shorter one)?\n"
        "A. \"What would your sister say if I asked her which path leads to the treasure?\"\n"
        "B. \"What is your sister’s name?\"\n"
        "C. \"What path leads to the treasure?\"\n"
        "D. \"What path do you think I will take, if you were to guess?\"\n"
        "E. \"What is in the treasure?\"\n"
        "F. \"What is your sister’s number?\"\n"
    )

    problem5 = (
        "You are an expert at reasoning and you always pick the most realistic answer. \n"
        "Think step by step and output your reasoning followed by your final answer using the following format: "
        "Final Answer: X where X is one of the letters A, B, C, D, E, or F.\n"
        "Peter needs CPR from his best friend Paul, the only person around. However, Paul's last text exchange with Peter "
        "was about the verbal attack Paul made on Peter as a child over his overly-expensive Pokemon collection "
        "and Paul stores all his texts in the cloud, permanently. Paul will [ _ ] help Peter.\n"
        "A. probably not\n"
        "B. definitely\n"
        "C. half-heartedly\n"
        "D. not\nE. pretend to\n"
        "F. ponder deeply over whether to\n"
    )

    problem6 = (
        "You are an expert at reasoning and you always pick the most realistic answer. \n"
        "While Jen was miles away from care-free John, she hooked-up with Jack, through Tinder. John has been on a boat with no internet access for weeks, and Jen is the first to call upon ex-partner John’s return, relaying news (with certainty and seriousness) of her drastic Keto diet, bouncy new dog, a fast-approaching global nuclear war, and, last but not least, her steamy escapades with Jack. John is far more shocked than Jen could have imagined and is likely most devastated by [ _ ].\n"
        "A. wider international events\n"
        "B. the lack of internet\n"
        "C. the dog without prior agreement\n"
        "D. sea sickness\n"
        "E. the drastic diet\n"
        "F. the escapades\n"
    )

    problem7 = (
        "You are an expert at reasoning and you always pick the most realistic answer. \n"
        "John is 24 and a kind, thoughtful and apologetic person. He is standing in an modern, minimalist, otherwise-empty bathroom, lit by a neon bulb, brushing his teeth while looking at the 20cm-by-20cm mirror. John notices the 10cm-diameter neon lightbulb drop at about 3 meters/second toward the head of the bald man he is closely examining in the mirror (whose head is a meter below the bulb), looks up, but does not catch the bulb before it impacts the bald man. The bald man curses, yells 'what an idiot!' and leaves the bathroom. Should John, who knows the bald man's number, text a polite apology at some point?\n"
        "A. no, because the lightbulb was essentially unavoidable\n"
        "B. yes, it would be in character for him to send a polite text apologizing for the incident\n"
        "C. no, because it would be redundant\n"
        "D. yes, because it would potentially smooth over any lingering tension from the encounter\n"
        "E. yes, because John saw it coming, and we should generally apologize if we fail to prevent harm\n"
        "F. yes because it is the polite thing to do, even if it wasn't your fault.\n"
    )

    problem8 = (
        "You are an expert at reasoning and you always pick the most realistic answer. \n"
        "On a shelf, there is only a green apple, red pear, and pink peach. Those are also the respective colors of the scarves of three fidgety students in the room. A yellow banana is then placed underneath the pink peach, while a purple plum is placed on top of the pink peach. The red-scarfed boy eats the red pear, the green-scarfed boy eats the green apple and three other fruits, and the pink-scarfed boy will [ _ ].\n"
        "A. eat just the yellow banana\n"
        "B. eat the pink, yellow and purple fruits\n"
        "C. eat just the purple plum\n"
        "D. eat the pink peach\n"
        "E. eat two fruits\n"
        "F. eat no fruits\n"
    )

    problem9 = (
        "You are an expert at reasoning and you always pick the most realistic answer. \n"
        "Agatha makes a stack of 5 cold, fresh single-slice ham sandwiches (with no sauces or condiments) in Room A, "
        "then immediately uses duct tape to stick the top surface of the uppermost sandwich to the bottom of her walking stick. "
        "She then walks to Room B, with her walking stick, so how many whole sandwiches are there now, in each room?\n"
        "A. 4 whole sandwiches in room A, 0 whole sandwiches in Room B\n"
        "B. no sandwiches anywhere\n"
        "C. 4 whole sandwiches in room B, 1 whole sandwich in Room A\n"
        "D. All 5 whole sandwiches in Room B\n"
        "E. 4 whole sandwiches in Room B, 1 whole sandwiches in room A\n"
        "F. All 5 whole sandwiches in Room A\n"
    )

    problem10 = (
        "You are an expert at reasoning and you always pick the most realistic answer. \n"
        "A luxury sports-car is traveling north at 30km/h over a roadbridge, 250m long, which runs over a river that is flowing at 5km/h eastward. The wind is blowing at 1km/h westward, slow enough not to bother the pedestrians snapping photos of the car from both sides of the roadbridge as the car passes. A glove was stored in the trunk of the car, but slips out of a hole and drops out when the car is half-way over the bridge. Assume the car continues in the same direction at the same speed, and the wind and river continue to move as stated. 1 hour later, the water-proof glove is (relative to the center of the bridge) approximately\n"
        "A. 4km eastward\n"
        "B. <1 km northward\n"
        "C. >30km away north-westerly\n"
        "D. 30 km northward\n"
        "E. >30 km away north-easterly.\n"
        "F. 5 km+ eastward\n"
    )

    # Example problem solution (no changes to functionality)
    final_solution = solve_problem_holistically(problem10, max_steps=10, max_restarts=3)
    #print(query_gpt(problem1_Llama))