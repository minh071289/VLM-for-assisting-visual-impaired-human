import json
from dataclasses import dataclass
from typing import List, Dict

from traitlets import Any

@dataclass
class POLMData:
    """POLM structure from bbox annotations"""
    object_type: str
    bbox: List[float]  # [x1, y1, x2, y2]
    confidence: float
    def to_text(self) -> str:
        return (
            f"[OBJ] {self.object_type} "
            f"({self.bbox[0]:.3f}, {self.bbox[1]:.3f}, "
            f"{self.bbox[2]:.3f}, {self.bbox[3]:.3f}) "
            f"conf: {self.confidence:.2f}"
        )

@dataclass
class GroundTruthData:
    """Ground truth for training"""
    location: str
    weather: str
    traffic: str
    scene: str
    instruction: str
    
    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps({
            'location': self.location,
            'weather': self.weather,
            'traffic': self.traffic,
            'scene': self.scene,
            'instruction': self.instruction
        }, ensure_ascii=False)

def construct_prompt(
    polm_list: List[POLMData],
    num_images: int = 1,
    metadata: Dict = None,  # ← THÊM
    model_architecture: str = 'qwen'    
) -> List[Dict[str, Any]]:
    """
    Construct model input messages (Updated for apply_chat_template)
    """
    
    polm_text = "\n".join([f"- {polm.to_text()}" for polm in polm_list])
    
    question = ""
    if metadata and metadata.get('QA') and metadata['QA'].get('Q'):
        question = metadata['QA']['Q']
    
    # Tạo nội dung text hướng dẫn (Prompt)
    text_content = f"""You are a navigation assistant for blind people.

Detected objects:
{polm_text}

Analyze: location, weather, traffic, scene → then give instruction.

Follow Chain-of-Thought reasoning:
1. Perception: Extract "location" (e.g., pedestrian_path, busy_street), "weather" (e.g., sunny, indoor), and "traffic" (e.g., high, moderate).
2. Comprehension: Synthesize details into the "scene".
3. Decision: Formulate the final "instruction"."""

    if question != "":
        text_content += f"\n\nQuestion: {question}"
        text_content += """

Format response:
<answer>{"location": "...", "weather": "...", "traffic": "...", "scene": "<concise visual summary, max 2 sentences>", "instruction": "<your answer to the question>"}</answer>

<answer>"""
    else:
        text_content += """

Format response:
<answer>{"location": "...", "weather": "...", "traffic": "...", "scene": "<concise visual summary, max 2 sentences>", "instruction": "<actionable alert and guidance>"}</answer>

<answer>"""
    content = []
    for _ in range(num_images):
        content.append({"type": "image"})
    content.append({"type": "text", "text": text_content})
    return [{"role": "user", "content": content}]
    
def map_metadata_to_ground_truth(metadata: Dict) -> GroundTruthData:
    """Map WAD metadata to ground truth format"""
    
    # Location mapping
    area_map = {
        'Pedestrian Path': 'pedestrian_path',
        'Road': 'road',
        'Corridor': 'corridor',
        'Busy Street': 'busy_street',
        'Shopping Mall': 'shopping_mall',
        'Bicycle Lane': 'bicycle_lane',
        'Restaurant': 'restaurant',
        'Other': 'other'
    }
    
    # Weather mapping
    weather_map = {
        'Sunny': 'sunny',
        'Overcast': 'overcast',
        'Cloudy': 'cloudy',
        'Night': 'night',
        'Indoor': 'indoor',
        'Other': 'other'
    }
    
    # Traffic mapping
    traffic_map = {
        'High': 'high',
        'Mid': 'moderate',
        'Low': 'low'
    }
    
    location = area_map.get(metadata.get('area_type', 'Other'), 'other')
    weather = weather_map.get(metadata.get('weather_condition', 'Other'), 'other')
    traffic = traffic_map.get(metadata.get('traffic_flow_rating', 'Low'), 'low')
    scene = metadata.get('summary', '')
    
    # Instruction: alter or QA answer
    if metadata.get('QA') and isinstance(metadata['QA'], dict):
        instruction = metadata['QA'].get('A', '')
    elif metadata.get('alter'):
        instruction = metadata['alter']
    else:
        instruction = ''
    
    return GroundTruthData(
        location=location,
        weather=weather,
        traffic=traffic,
        scene=scene,
        instruction=instruction
    )