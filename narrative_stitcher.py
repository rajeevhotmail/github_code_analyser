from openai import OpenAI

class NarrativeStitcher:
    def __init__(self, model="gpt-4o"):
        self.model = model
        self.client = OpenAI()  # uses OPENAI_API_KEY from env

    def stitch_narrative(self, repo_name: str, role: str, qa_pairs: list[dict]) -> str:
        points = "\n".join(
            f"- {qa['answer'].strip()}" for qa in qa_pairs
        )

        prompt = (
            f"You are a technical writer preparing a report on the repository '{repo_name}', "
            f"from the perspective of a {role}. Below are key findings extracted from a Q&A process.\n\n"
            "Please write a cohesive, flowing narrative. Begin with a short summary of the project. "
            "Then elaborate each point as a paragraph. Avoid listing questions, and do not use section headers. "
            "Just write in continuous, natural prose.\n\n"
            f"At the end of your response, include a section titled 'Key Findings' with 5–7 bullet points summarizing the most important insights. "
            f"These should be suitable for someone in the role of a {role}.\n\n"
            f"Here are the key points:\n{points}"
        )

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": "You are a helpful technical writer."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
            max_tokens=2048
        )

        return response.choices[0].message.content.strip()

# Inside narrative_stitcher.py, after the class

def extract_narrative_and_key_findings(full_text: str):
    split_marker = "Key Findings"
    if split_marker in full_text:
        narrative, key_section = full_text.split(split_marker, 1)
        key_points = key_section.strip().split('\n')
        key_findings = [
            line.lstrip("•").lstrip("-").strip()
            for line in key_points
            if line.strip()
        ]
        return narrative.strip(), key_findings
    else:
        return full_text.strip(), []
