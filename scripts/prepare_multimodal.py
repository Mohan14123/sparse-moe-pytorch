#!/usr/bin/env python3
"""
Multimodal dataset generator for Sparse Mixture-of-Experts training.

Generates synthetic data for 10 tasks across 3 modalities:
  - Text tasks (0-4): Code, Maths, Science, Law, Story
  - Image tasks (5-7): Classification, Reconstruction, Edge Detection
  - Vision-Language tasks (8-9): Image Captioning, VQA

Usage:
    python scripts/prepare_multimodal.py --samples_per_task 1000
"""

import argparse
import os
import random
import math

import numpy as np

try:
    from PIL import Image, ImageDraw
except ImportError:
    raise ImportError("Pillow is required. Install with: pip install Pillow")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

IMG_SIZE = 32
SHAPE_NAMES = ["circle", "square", "triangle", "cross", "line"]  # labels 0-4

COLOR_PALETTE = [
    (255, 0, 0),      # red
    (0, 200, 0),      # green
    (0, 0, 255),      # blue
    (255, 255, 0),    # yellow
    (255, 0, 255),    # magenta
    (0, 255, 255),    # cyan
    (255, 128, 0),    # orange
    (128, 0, 255),    # purple
    (0, 200, 200),    # teal
    (255, 255, 255),  # white
]

COLOR_NAMES = [
    "red", "green", "blue", "yellow", "magenta",
    "cyan", "orange", "purple", "teal", "white",
]

POSITION_NAMES = ["in the center", "on the left", "on the right", "at the top", "at the bottom"]

# ---------------------------------------------------------------------------
# Text-task template pools
# ---------------------------------------------------------------------------

CODE_TEMPLATES = [
    # 1 – simple function
    '''def add(a, b):
    """Return the sum of a and b."""
    return a + b

def subtract(a, b):
    """Return the difference of a and b."""
    return a - b
''',
    # 2 – class definition
    '''class NeuralNetwork(nn.Module):
    """A simple feedforward neural network."""
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x
''',
    # 3 – loops and conditionals
    '''# Nested loop example
results = []
for i in range(10):
    for j in range(i, 20):
        if (i + j) % 3 == 0:
            results.append((i, j, i * j))
        elif (i - j) % 2 == 0:
            results.append((i, j, i + j))
print(f"Total qualifying pairs: {len(results)}")
''',
    # 4 – imports and tensor ops
    '''import torch
import torch.nn.functional as F
import numpy as np

x = torch.randn(64, 128)
weights = torch.randn(128, 256)
output = F.softmax(x @ weights, dim=-1)
print(f"Output shape: {output.shape}, sum per row: {output.sum(dim=-1)[:5]}")
''',
    # 5 – decorator
    '''import functools
import time

def timer_decorator(func):
    """Measure execution time of the decorated function."""
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        result = func(*args, **kwargs)
        elapsed = time.perf_counter() - start
        print(f"{func.__name__} took {elapsed:.4f}s")
        return result
    return wrapper

@timer_decorator
def heavy_computation(n):
    return sum(i * i for i in range(n))
''',
    # 6 – list comprehension and generators
    '''# Advanced list comprehensions
squares = [x ** 2 for x in range(50) if x % 3 != 0]
nested = [[row * col for col in range(1, 6)] for row in range(1, 11)]
flat = [item for sublist in nested for item in sublist]

# Generator expression for memory efficiency
def fibonacci_gen(limit):
    a, b = 0, 1
    while a < limit:
        yield a
        a, b = b, a + b

fibs = list(fibonacci_gen(10000))
print(f"Fibonacci numbers below 10000: {len(fibs)}")
''',
    # 7 – context managers
    '''import contextlib
import tempfile
import json

@contextlib.contextmanager
def managed_resource(name):
    print(f"Acquiring resource: {name}")
    resource = {"name": name, "active": True}
    try:
        yield resource
    finally:
        resource["active"] = False
        print(f"Released resource: {name}")

with managed_resource("database_connection") as conn:
    print(f"Using connection: {conn}")

with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
    json.dump({"key": "value", "items": [1, 2, 3]}, f)
    print(f"Wrote temp file: {f.name}")
''',
    # 8 – error handling
    '''class ValidationError(Exception):
    """Custom validation error for data processing."""
    def __init__(self, field, message):
        self.field = field
        self.message = message
        super().__init__(f"Validation failed for '{field}': {message}")

def validate_input(data):
    if not isinstance(data, dict):
        raise ValidationError("data", "Expected a dictionary")
    required = ["name", "age", "email"]
    for field in required:
        if field not in data:
            raise ValidationError(field, f"Missing required field")
        if not data[field]:
            raise ValidationError(field, "Field cannot be empty")
    if not isinstance(data["age"], int) or data["age"] < 0:
        raise ValidationError("age", "Must be a non-negative integer")
    return True

try:
    validate_input({"name": "Alice", "age": 30, "email": "alice@example.com"})
    print("Validation passed!")
except ValidationError as e:
    print(f"Error: {e}")
''',
    # 9 – generators and itertools
    '''import itertools
from collections import Counter

def sliding_window(iterable, n):
    """Generate overlapping windows of size n."""
    it = iter(iterable)
    window = []
    for _ in range(n):
        window.append(next(it))
    yield tuple(window)
    for item in it:
        window.pop(0)
        window.append(item)
        yield tuple(window)

data = list(range(20))
windows = list(sliding_window(data, 5))
print(f"Number of windows: {len(windows)}")

# Combinatorial generation
combos = list(itertools.combinations("ABCDE", 3))
perms = list(itertools.permutations("XYZ", 2))
print(f"Combinations: {len(combos)}, Permutations: {len(perms)}")
''',
    # 10 – dataclass
    '''from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class Experiment:
    name: str
    learning_rate: float
    batch_size: int
    epochs: int
    metrics: List[float] = field(default_factory=list)
    description: Optional[str] = None

    def add_metric(self, value: float):
        self.metrics.append(value)

    @property
    def best_metric(self) -> Optional[float]:
        return max(self.metrics) if self.metrics else None

    def summary(self) -> str:
        return (f"Experiment '{self.name}': lr={self.learning_rate}, "
                f"bs={self.batch_size}, epochs={self.epochs}, "
                f"best={self.best_metric}")

exp = Experiment("moe_v1", 1e-4, 32, 100)
for val in [0.85, 0.88, 0.91, 0.89, 0.93]:
    exp.add_metric(val)
print(exp.summary())
''',
    # 11 – async
    '''import asyncio

async def fetch_data(url, delay):
    """Simulate an asynchronous data fetch."""
    print(f"Fetching {url}...")
    await asyncio.sleep(delay)
    return {"url": url, "status": 200, "data": f"Response from {url}"}

async def process_batch(urls):
    tasks = [fetch_data(url, random.uniform(0.1, 0.5)) for url in urls]
    results = await asyncio.gather(*tasks)
    for result in results:
        print(f"  Got {result['status']} from {result['url']}")
    return results

urls = [f"https://api.example.com/data/{i}" for i in range(5)]
# asyncio.run(process_batch(urls))
''',
    # 12 – dictionary and set operations
    '''# Advanced dictionary operations
inventory = {
    "apples": 50, "bananas": 30, "oranges": 45,
    "grapes": 60, "mangoes": 25, "pears": 35,
}

# Filter, transform, merge
expensive = {k: v for k, v in inventory.items() if v > 35}
doubled = {k: v * 2 for k, v in inventory.items()}
merged = {**inventory, **{"kiwis": 20, "apples": 55}}

# Set operations on keys
set_a = set(inventory.keys())
set_b = {"apples", "kiwis", "peaches", "grapes"}
common = set_a & set_b
unique_to_a = set_a - set_b
print(f"Common items: {common}")
print(f"Only in inventory: {unique_to_a}")
''',
    # 13 – recursion
    '''def merge_sort(arr):
    """Implement merge sort algorithm."""
    if len(arr) <= 1:
        return arr
    mid = len(arr) // 2
    left = merge_sort(arr[:mid])
    right = merge_sort(arr[mid:])
    return merge(left, right)

def merge(left, right):
    result = []
    i = j = 0
    while i < len(left) and j < len(right):
        if left[i] <= right[j]:
            result.append(left[i])
            i += 1
        else:
            result.append(right[j])
            j += 1
    result.extend(left[i:])
    result.extend(right[j:])
    return result

data = [38, 27, 43, 3, 9, 82, 10]
sorted_data = merge_sort(data)
print(f"Sorted: {sorted_data}")
''',
    # 14 – type hints and protocols
    '''from typing import Protocol, runtime_checkable, TypeVar, Generic

T = TypeVar("T")

@runtime_checkable
class Serializable(Protocol):
    def to_dict(self) -> dict: ...
    def from_dict(cls, data: dict) -> "Serializable": ...

class Repository(Generic[T]):
    def __init__(self):
        self._store: dict[str, T] = {}

    def add(self, key: str, item: T) -> None:
        self._store[key] = item

    def get(self, key: str) -> T | None:
        return self._store.get(key)

    def list_all(self) -> list[T]:
        return list(self._store.values())

    def count(self) -> int:
        return len(self._store)

repo: Repository[str] = Repository()
repo.add("greeting", "hello world")
print(f"Items in repo: {repo.count()}")
''',
    # 15 – file I/O and CSV
    '''import csv
import io

def write_csv_report(records, headers):
    """Generate a CSV report from a list of dictionaries."""
    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=headers)
    writer.writeheader()
    for record in records:
        writer.writerow(record)
    return output.getvalue()

records = [
    {"name": "Model_A", "accuracy": 0.95, "loss": 0.12, "params": "10M"},
    {"name": "Model_B", "accuracy": 0.93, "loss": 0.15, "params": "5M"},
    {"name": "Model_C", "accuracy": 0.97, "loss": 0.08, "params": "50M"},
]
report = write_csv_report(records, ["name", "accuracy", "loss", "params"])
print(report)
''',
]

MATH_TEMPLATES = [
    "Calculate the result of {a} {op} {b}. The answer is {res}.",
    "If x = {a} and y = {b}, then x {op} y equals {res}.",
    "{a} {op} {b} = {res}.",
    "Solve for z: z = {a} {op} {b}. Therefore z = {res}.",
    "The sum of {a} and {b} is {res}." if None else "Compute {a} {op} {b} to get {res}.",
    "Find the value: {a} {op} {b}. Answer: {res}.",
    "Given the expression {a} {op} {b}, evaluate it to obtain {res}.",
    "A shopkeeper has {a} items and buys {b} more. The total is {a} + {b} = {res}.",
    "What is {a} {op} {b}? The result is {res}.",
    "Evaluate the following: {a} {op} {b} = {res}. This can be verified by inspection.",
]

MATH_WORD_PROBLEMS = [
    "A train travels at {a} km/h for {b} hours. The distance covered is {a} × {b} = {res} km.",
    "If a rectangle has length {a} cm and width {b} cm, its area is {a} × {b} = {res} square cm.",
    "A student scored {a} marks in the first test and {b} marks in the second. The total score is {res}.",
    "There are {a} red balls and {b} blue balls in a bag. The total number of balls is {res}.",
    "A factory produces {a} units per hour. In {b} hours, it produces {a} × {b} = {res} units.",
    "The perimeter of a square with side {a} is 4 × {a} = {res}.",
    "If you divide {a} cookies equally among {b} children, each child gets {a} ÷ {b} = {res} cookies.",
    "A library has {a} books on the first floor and {b} books on the second floor, totaling {res} books.",
]

MATH_ALGEBRA = [
    "Let f(x) = {a}x + {b}. Then f(1) = {a}(1) + {b} = {res}.",
    "Solve: {a}x = {res}. Therefore x = {res} / {a} = {b}.",
    "The quadratic equation x² - {a}x + {b} = 0 has discriminant D = {a}² - 4({b}).",
    "Given the sequence a_n = {a}n + {b}, the first five terms are: {terms}.",
    "If log base 10 of {a} is approximately {log_val}, then 10^{log_val} ≈ {a}.",
]

SCIENCE_PARAGRAPHS = [
    # Physics
    ("Newton's second law of motion states that the force acting on an object is equal to "
     "the mass of that object multiplied by its acceleration, expressed mathematically as "
     "F = ma. This fundamental principle governs the motion of all macroscopic objects in "
     "our universe. When a net force is applied to a body, it produces a proportional "
     "acceleration inversely related to the body's mass. This law forms the cornerstone "
     "of classical mechanics and has been extensively validated through centuries of "
     "experimental observation. Applications range from calculating rocket trajectories "
     "to designing automotive safety systems."),

    ("The theory of general relativity, proposed by Albert Einstein in 1915, describes "
     "gravity not as a force but as a curvature of spacetime caused by mass and energy. "
     "Massive objects like stars and planets create distortions in the fabric of spacetime, "
     "and other objects follow curved paths through this warped geometry. This elegant "
     "framework has been confirmed by numerous experiments, including the observation of "
     "gravitational lensing, the precession of Mercury's orbit, and the detection of "
     "gravitational waves by LIGO in 2015."),

    ("Thermodynamics is the branch of physics that deals with heat, work, and the "
     "relationships between them. The first law, also known as the law of conservation "
     "of energy, states that energy cannot be created or destroyed, only transformed from "
     "one form to another. The second law introduces the concept of entropy, which tends "
     "to increase in any isolated system. The third law states that as temperature "
     "approaches absolute zero, the entropy of a perfect crystal approaches zero. These "
     "laws have profound implications for engineering, chemistry, and cosmology."),

    # Biology
    ("Photosynthesis is the process by which green plants, algae, and certain bacteria "
     "convert light energy, usually from the sun, into chemical energy stored in glucose "
     "molecules. This process occurs primarily in the chloroplasts of plant cells, where "
     "chlorophyll pigments absorb light energy. The overall equation is 6CO₂ + 6H₂O + "
     "light energy → C₆H₁₂O₆ + 6O₂. Photosynthesis is divided into two stages: the "
     "light-dependent reactions, which occur in the thylakoid membranes, and the Calvin "
     "cycle, which takes place in the stroma of the chloroplast."),

    ("DNA (deoxyribonucleic acid) is the molecule that carries the genetic instructions "
     "for the development, functioning, growth, and reproduction of all known organisms "
     "and many viruses. It consists of two polynucleotide chains that coil around each "
     "other to form a double helix structure. The nucleotide bases adenine, thymine, "
     "guanine, and cytosine pair specifically (A-T and G-C) through hydrogen bonds. "
     "The sequence of these base pairs encodes the information needed to build proteins, "
     "which perform the vast majority of functions within living organisms."),

    ("The human immune system is a complex network of cells, tissues, and organs that "
     "work together to defend the body against harmful pathogens including bacteria, "
     "viruses, fungi, and parasites. It consists of two main components: the innate "
     "immune system, which provides immediate but non-specific defense, and the adaptive "
     "immune system, which develops targeted responses to specific pathogens. Key players "
     "include white blood cells such as T cells, B cells, and natural killer cells, "
     "along with antibodies and complement proteins."),

    # Chemistry
    ("The periodic table of elements organizes all known chemical elements by their "
     "atomic number, electron configuration, and recurring chemical properties. Elements "
     "are arranged in order of increasing atomic number into rows called periods and "
     "columns called groups. Elements in the same group share similar chemical properties "
     "because they have the same number of valence electrons. The periodic table was "
     "first proposed by Dmitri Mendeleev in 1869 and has since become one of the most "
     "important tools in chemistry, enabling scientists to predict the properties of "
     "undiscovered elements and understand chemical bonding patterns."),

    ("Chemical bonding refers to the attractive forces that hold atoms together in "
     "molecules and compounds. The three primary types of chemical bonds are ionic bonds, "
     "formed by the transfer of electrons between atoms; covalent bonds, formed by the "
     "sharing of electron pairs; and metallic bonds, found in metals where electrons are "
     "delocalized across a lattice of positive ions. The type of bonding significantly "
     "affects the physical and chemical properties of substances, including melting point, "
     "electrical conductivity, solubility, and hardness."),

    # Astronomy
    ("The Milky Way galaxy is a barred spiral galaxy containing an estimated 100 to 400 "
     "billion stars and at least that many planets. Our solar system is located in one of "
     "the spiral arms, approximately 26,000 light-years from the galactic center. The "
     "galaxy has a diameter of about 100,000 light-years and is estimated to be about "
     "13.6 billion years old. At its center lies a supermassive black hole known as "
     "Sagittarius A*, which has a mass approximately 4 million times that of our Sun."),

    ("Black holes are regions of spacetime where gravity is so strong that nothing, not "
     "even light or other electromagnetic waves, has enough energy to escape the event "
     "horizon. They are predicted by Einstein's general theory of relativity. Stellar "
     "black holes form when massive stars exhaust their nuclear fuel and collapse under "
     "their own gravity. Supermassive black holes, found at the centers of galaxies, can "
     "have masses billions of times that of our Sun. The first direct image of a black "
     "hole's shadow was captured by the Event Horizon Telescope in 2019."),

    # Earth Science
    ("Plate tectonics is the scientific theory describing the large-scale motion of "
     "Earth's lithosphere, which is divided into several major and minor tectonic plates. "
     "These plates float on the semi-fluid asthenosphere beneath them and interact at "
     "their boundaries through convergence, divergence, and transform motion. Plate "
     "tectonics explains the distribution of earthquakes and volcanoes, the formation of "
     "mountain ranges, and the opening and closing of ocean basins over geological time."),

    ("The water cycle, also known as the hydrological cycle, describes the continuous "
     "movement of water within the Earth and atmosphere. It involves evaporation from "
     "surface water bodies, transpiration from plants, condensation of water vapor into "
     "clouds, precipitation as rain or snow, and collection in rivers, lakes, and oceans. "
     "Groundwater seepage and runoff complete the cycle. Understanding the water cycle is "
     "essential for managing freshwater resources and predicting weather patterns."),
]

LAW_PARAGRAPHS = [
    # Contract Law
    ("Whereas the Party of the First Part, hereinafter referred to as the 'Licensor,' "
     "hereby grants to the Party of the Second Part, hereinafter referred to as the "
     "'Licensee,' a non-exclusive, non-transferable license to use the intellectual "
     "property described in Schedule A, subject to the terms and conditions set forth "
     "herein. This agreement shall be effective as of the date of last signature and "
     "shall remain in force for a period of five (5) years unless terminated earlier "
     "in accordance with Section 8 of this agreement. The Licensee agrees to pay the "
     "Licensor a royalty fee as specified in Schedule B."),

    ("The parties agree that any breach of this contract shall entitle the non-breaching "
     "party to seek specific performance or monetary damages as determined by a court of "
     "competent jurisdiction. In no event shall either party be liable for consequential, "
     "incidental, or punitive damages arising from the performance or non-performance of "
     "obligations under this agreement, except in cases of willful misconduct or gross "
     "negligence. The prevailing party in any legal action shall be entitled to recover "
     "reasonable attorney's fees and court costs from the other party."),

    # Constitutional Law
    ("Under the provisions of Section 14 of the Constitution, every person is guaranteed "
     "the right to equality before the law and equal protection of the laws within the "
     "territory of the nation. No person shall be discriminated against on grounds of "
     "religion, race, caste, sex, or place of birth. This fundamental right is subject "
     "to reasonable restrictions that may be imposed by the State in the interests of "
     "public order, morality, and the sovereignty and integrity of the nation. The Supreme "
     "Court has consistently upheld these protections through landmark judicial decisions."),

    ("The doctrine of separation of powers divides the government into three distinct "
     "branches: the legislative, the executive, and the judiciary. Each branch operates "
     "independently with its own powers and responsibilities, and no branch may exercise "
     "the powers belonging to another. This system of checks and balances ensures that no "
     "single branch becomes too powerful and that the rights of citizens are protected "
     "against potential abuse of governmental authority. The judiciary serves as the "
     "guardian of the constitution and has the power of judicial review."),

    # Criminal Law
    ("The defendant is charged with violation of Section 302 of the Penal Code, which "
     "pertains to the offense of culpable homicide amounting to murder. The prosecution "
     "must establish beyond a reasonable doubt that the accused acted with the intention "
     "of causing death, or with the intention of causing such bodily injury as the "
     "accused knew to be likely to cause the death of the person to whom the harm was "
     "caused. The burden of proof rests entirely upon the prosecution, and the accused "
     "shall be presumed innocent until proven guilty in a court of law."),

    ("In criminal proceedings, the right to a fair trial encompasses several fundamental "
     "guarantees including the right to be informed of the nature and cause of the "
     "accusation, the right to counsel, the right to examine witnesses, and the right "
     "against self-incrimination. The accused must be given adequate time and facilities "
     "to prepare a defense. Any confession obtained through coercion, inducement, or "
     "threat shall be deemed inadmissible as evidence. The trial must be conducted "
     "expeditiously and without unreasonable delay."),

    # Intellectual Property
    ("Patent rights shall be granted to any person who invents or discovers any new and "
     "useful process, machine, manufacture, or composition of matter, or any new and "
     "useful improvement thereof, subject to the conditions and requirements of this "
     "title. The patent holder shall have the exclusive right to make, use, offer for "
     "sale, sell, and import the patented invention for a period of twenty (20) years "
     "from the date of filing of the patent application. Any infringement of these "
     "rights shall subject the infringer to civil liability and injunctive relief."),

    ("Copyright protection subsists in original works of authorship fixed in any "
     "tangible medium of expression, including literary, dramatic, musical, and artistic "
     "works, as well as computer programs, compilations, and architectural works. The "
     "copyright owner has the exclusive right to reproduce the work, prepare derivative "
     "works, distribute copies, perform the work publicly, and display the work publicly. "
     "Copyright protection extends for the life of the author plus seventy (70) years. "
     "Fair use provisions allow limited use for purposes such as criticism, commentary, "
     "news reporting, teaching, and research."),

    # Regulatory Compliance
    ("In accordance with Regulation 15(2) of the Data Protection Act, every data "
     "controller must implement appropriate technical and organizational measures to "
     "ensure a level of security appropriate to the risk presented by the processing of "
     "personal data. Such measures shall include, as appropriate, the pseudonymization "
     "and encryption of personal data, the ability to ensure the ongoing confidentiality, "
     "integrity, availability, and resilience of processing systems and services, and "
     "the ability to restore the availability and access to personal data in a timely "
     "manner in the event of a physical or technical incident."),

    ("The regulatory authority shall have the power to impose administrative fines of "
     "up to four percent (4%) of the total worldwide annual turnover of the preceding "
     "financial year, or twenty million euros, whichever is higher, for infringements "
     "of the basic principles for processing, including conditions for consent, the "
     "data subjects' rights, and the conditions for transfers of personal data to "
     "third countries or international organizations. The authority shall ensure that "
     "the imposition of fines is effective, proportionate, and dissuasive in each "
     "individual case."),

    # Corporate Law
    ("The Board of Directors shall have the authority to declare and pay dividends to "
     "shareholders out of the surplus profits of the company, subject to the provisions "
     "of the Companies Act and the Articles of Association. No dividend shall be paid "
     "except out of profits of the company or out of monies provided by the central or "
     "state government for the payment of such dividend in pursuance of a guarantee "
     "given by the government. The company shall maintain adequate reserves before "
     "declaring any dividend as required by applicable regulations."),
]

STORY_CHARACTERS = [
    "Elena", "Marcus", "Aria", "Kai", "Sofia", "Jasper", "Luna", "Rowan",
    "Ivy", "Finn", "Nadia", "Oscar", "Zara", "Theo", "Maya", "Leo",
]

STORY_SETTINGS = [
    "a fog-covered coastal village", "an ancient library hidden beneath the mountains",
    "a bustling space station orbiting Mars", "a quiet forest clearing at twilight",
    "a grand Victorian mansion", "a lively street market in a foreign city",
    "a remote research outpost in Antarctica", "a floating city above the clouds",
    "a medieval castle surrounded by enchanted woods", "a neon-lit cyberpunk metropolis",
    "a tranquil monastery on a mountain peak", "a submarine exploring the ocean depths",
]

STORY_PLOTS = [
    "discovered a mysterious map that led to an ancient treasure",
    "received a cryptic letter from someone they had never met",
    "stumbled upon a hidden doorway that led to another dimension",
    "was chosen to lead an expedition into uncharted territory",
    "found an old journal that revealed a family secret",
    "encountered a stranger who claimed to know their future",
    "witnessed an extraordinary event that defied all scientific explanation",
    "uncovered a conspiracy that threatened the entire community",
    "was given a choice that would change the course of history",
    "learned to master an extraordinary ability they never knew they had",
]

STORY_MIDDLES = [
    ("Along the way, challenges arose that tested their resolve. The path was fraught "
     "with obstacles—some physical, others moral—that demanded courage and ingenuity. "
     "Allies appeared in unexpected places, while betrayals came from those once trusted. "
     "Each decision carried weight, and the consequences rippled through the narrative "
     "like stones thrown into still water."),
    ("The days that followed were filled with discovery and wonder. Every corner turned "
     "revealed new mysteries waiting to be unraveled. The landscape shifted from familiar "
     "to fantastical, and the rules that once governed reality seemed to bend and warp. "
     "Through determination and a willingness to embrace the unknown, progress was made "
     "step by careful step."),
    ("Tensions rose as competing factions vied for control of the situation. Negotiations "
     "were tense, filled with veiled threats and hidden agendas. Trust was a currency in "
     "short supply, and every alliance came with strings attached. Yet amidst the chaos, "
     "a clarity of purpose emerged that guided their actions forward."),
    ("Time seemed to move differently here, stretching and compressing in ways that "
     "defied understanding. Memories blurred with present experiences, creating a "
     "tapestry of moments both lived and imagined. The boundary between what was real "
     "and what was possible grew thinner with each passing hour."),
]

STORY_ENDINGS = [
    ("In the end, the journey had transformed them in ways they never expected. The "
     "world looked different now—richer, more complex, full of hidden connections. "
     "Standing at the threshold of a new chapter, they understood that every ending "
     "was merely the beginning of something else, something greater than what came before."),
    ("As the final pieces fell into place, a sense of profound peace settled over "
     "everything. The struggles had been worth it—each hardship a lesson, each failure "
     "a stepping stone. The story would be told and retold, growing in the telling, "
     "becoming legend in the memories of those who lived through it."),
    ("Looking back on everything that had transpired, there was no regret, only "
     "gratitude for the experience. The answers they had sought were found not in "
     "some grand revelation but in the quiet moments of connection and understanding "
     "that happened along the way. And so life continued, forever changed."),
]


# ---------------------------------------------------------------------------
# Helper: image generation
# ---------------------------------------------------------------------------

def draw_shape(shape_id, color, img_size=IMG_SIZE):
    """Draw a shape on a black background. Returns (PIL.Image, shape_info_dict)."""
    img = Image.new("RGB", (img_size, img_size), (0, 0, 0))
    draw = ImageDraw.Draw(img)

    margin = 4
    cx = random.randint(margin + 4, img_size - margin - 4)
    cy = random.randint(margin + 4, img_size - margin - 4)
    size = random.randint(4, min(cx, cy, img_size - cx, img_size - cy) - 1)
    size = max(size, 3)

    # Determine position name for captions
    if cx < img_size // 3:
        pos_name = "on the left"
    elif cx > 2 * img_size // 3:
        pos_name = "on the right"
    elif cy < img_size // 3:
        pos_name = "at the top"
    elif cy > 2 * img_size // 3:
        pos_name = "at the bottom"
    else:
        pos_name = "in the center"

    if shape_id == 0:  # circle
        draw.ellipse([cx - size, cy - size, cx + size, cy + size], fill=color)
    elif shape_id == 1:  # square
        draw.rectangle([cx - size, cy - size, cx + size, cy + size], fill=color)
    elif shape_id == 2:  # triangle
        pts = [
            (cx, cy - size),
            (cx - size, cy + size),
            (cx + size, cy + size),
        ]
        draw.polygon(pts, fill=color)
    elif shape_id == 3:  # cross
        arm = max(size // 3, 1)
        draw.rectangle([cx - arm, cy - size, cx + arm, cy + size], fill=color)
        draw.rectangle([cx - size, cy - arm, cx + size, cy + arm], fill=color)
    elif shape_id == 4:  # line
        angle = random.uniform(0, math.pi)
        dx = int(size * math.cos(angle))
        dy = int(size * math.sin(angle))
        draw.line([(cx - dx, cy - dy), (cx + dx, cy + dy)], fill=color, width=2)

    info = {
        "shape_id": shape_id,
        "shape_name": SHAPE_NAMES[shape_id],
        "color_name": COLOR_NAMES[COLOR_PALETTE.index(color)],
        "pos_name": pos_name,
    }
    return img, info


def compute_edge_map(img):
    """Compute a binary edge map from an image using outline approach."""
    arr = np.array(img).astype(np.float32)
    # Convert to grayscale
    gray = arr.mean(axis=2)
    # Create mask of non-black pixels
    mask = (gray > 10).astype(np.float32)

    # Simple edge detection: pixel is edge if it is part of the shape
    # but at least one of its 4-neighbors is not
    edge = np.zeros_like(mask)
    # pad mask
    padded = np.pad(mask, 1, mode='constant', constant_values=0)
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        shifted = padded[1 + dy:1 + dy + IMG_SIZE, 1 + dx:1 + dx + IMG_SIZE]
        edge += (mask > 0) * (shifted == 0)
    edge = (edge > 0).astype(np.float32)
    return edge


def pil_to_array(img):
    """Convert PIL Image (RGB) to float32 array [3, H, W] in [0, 1]."""
    arr = np.array(img).astype(np.float32) / 255.0  # [H, W, 3]
    arr = arr.transpose(2, 0, 1)  # [3, H, W]
    return arr


# ---------------------------------------------------------------------------
# Text generators
# ---------------------------------------------------------------------------

def generate_code_text():
    """Generate a code text sample from templates."""
    n = random.randint(5, 10)
    snippets = random.choices(CODE_TEMPLATES, k=n)
    return "\n\n".join(snippets)


def generate_math_text():
    """Generate a math text sample from templates."""
    parts = []
    operations = ['+', '-', '*', '/']
    n = random.randint(15, 25)
    for _ in range(n):
        op = random.choice(operations)
        a = random.randint(1, 1000)
        b = random.randint(1, 1000)
        if op == '+':
            res = a + b
        elif op == '-':
            res = a - b
        elif op == '*':
            a = random.randint(1, 100)
            b = random.randint(1, 100)
            res = a * b
        else:
            b = random.randint(1, 100)
            res_val = random.randint(1, 100)
            a = b * res_val
            res = res_val

        # Pick a template type
        r = random.random()
        if r < 0.4:
            tmpl = random.choice(MATH_TEMPLATES)
            parts.append(tmpl.format(a=a, b=b, op=op, res=res))
        elif r < 0.7:
            tmpl = random.choice(MATH_WORD_PROBLEMS)
            parts.append(tmpl.format(a=a, b=b, res=a + b if 'sum' in tmpl.lower() or 'total' in tmpl.lower() else a * b))
        else:
            # Algebra-style
            terms = ", ".join(str(a * i + b) for i in range(1, 6))
            log_val = f"{math.log10(max(a, 1)):.2f}"
            tmpl = random.choice(MATH_ALGEBRA)
            parts.append(tmpl.format(a=a, b=b, res=a + b, terms=terms, log_val=log_val))

    return "\n".join(parts)


def generate_science_text():
    """Generate a science text sample from paragraphs."""
    n = random.randint(5, 8)
    paragraphs = random.choices(SCIENCE_PARAGRAPHS, k=n)
    return "\n\n".join(paragraphs)


def generate_law_text():
    """Generate a legal text sample from paragraphs."""
    n = random.randint(5, 8)
    paragraphs = random.choices(LAW_PARAGRAPHS, k=n)
    return "\n\n".join(paragraphs)


def generate_story_text(story_source_texts=None):
    """Generate a creative narrative.

    If *story_source_texts* is a non-empty list, randomly sample from it;
    otherwise generate a synthetic story.
    """
    if story_source_texts:
        return random.choice(story_source_texts)

    # Synthetic story generation
    char = random.choice(STORY_CHARACTERS)
    char2 = random.choice([c for c in STORY_CHARACTERS if c != char])
    setting = random.choice(STORY_SETTINGS)
    plot = random.choice(STORY_PLOTS)

    opening = (
        f"In {setting}, {char} lived a quiet but purposeful life. "
        f"Everything changed the day {char} {plot}. "
        f"It was an ordinary morning when the extraordinary began—sunlight filtering "
        f"through the windows carried an unusual warmth, and the air hummed with "
        f"an energy that was impossible to ignore. {char} knew, even before "
        f"understanding why, that nothing would ever be the same."
    )

    companion = (
        f"It wasn't long before {char2} entered the picture. "
        f"They had met years ago under entirely different circumstances, but fate—or "
        f"perhaps something more deliberate—had drawn them together once more. "
        f"{char2} brought skills and perspectives that complemented {char}'s own, "
        f"and together they formed an unlikely but formidable partnership."
    )

    middle = random.choice(STORY_MIDDLES)
    ending = random.choice(STORY_ENDINGS)

    paragraphs = [opening, companion, middle, ending]
    # Optionally add an extra middle paragraph
    if random.random() < 0.5:
        extra = random.choice([m for m in STORY_MIDDLES if m != middle])
        paragraphs.insert(3, extra)

    return "\n\n".join(paragraphs)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generate a multimodal dataset for Sparse MoE training."
    )
    parser.add_argument(
        "--samples_per_task", type=int, default=1000,
        help="Number of samples to generate per task (default: 1000)."
    )
    parser.add_argument(
        "--output", type=str, default="datasets/multimodal_train.npz",
        help="Output file path (default: datasets/multimodal_train.npz)."
    )
    parser.add_argument(
        "--source_story_csv", type=str, default="datasets/train.csv",
        help="Optional CSV with a 'text' column to sample story data from."
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)."
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    spt = args.samples_per_task
    total = 10 * spt
    print(f"Generating multimodal dataset: {spt} samples × 10 tasks = {total} total")

    # Pre-allocate arrays
    texts = np.empty(total, dtype=object)
    task_ids = np.zeros(total, dtype=np.int32)
    images = np.zeros((total, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    noisy_images = np.zeros((total, 3, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    edge_maps = np.zeros((total, 1, IMG_SIZE, IMG_SIZE), dtype=np.float32)
    class_labels = np.zeros(total, dtype=np.int32)
    vqa_answers = np.zeros(total, dtype=np.int32)

    idx = 0  # running sample index

    # ------------------------------------------------------------------
    # Try loading story source
    # ------------------------------------------------------------------
    story_source_texts = None
    if os.path.exists(args.source_story_csv):
        try:
            import pandas as pd
            print(f"Loading story source from {args.source_story_csv} ...")
            df = pd.read_csv(args.source_story_csv, usecols=["text"])
            story_source_texts = df["text"].dropna().tolist()
            # Filter for reasonably long texts (200+ words)
            story_source_texts = [t for t in story_source_texts if len(t.split()) >= 200]
            if len(story_source_texts) < spt:
                print(f"  Only {len(story_source_texts)} long texts found; "
                      f"will supplement with synthetic stories.")
            else:
                print(f"  Found {len(story_source_texts)} qualifying stories.")
        except Exception as e:
            print(f"  Warning: Could not load story CSV: {e}. Using synthetic stories.")
            story_source_texts = None

    # ------------------------------------------------------------------
    # Task 0: Code
    # ------------------------------------------------------------------
    print("Task 0 — Code ...")
    for i in range(spt):
        texts[idx] = generate_code_text()
        task_ids[idx] = 0
        idx += 1

    # ------------------------------------------------------------------
    # Task 1: Maths
    # ------------------------------------------------------------------
    print("Task 1 — Maths ...")
    for i in range(spt):
        texts[idx] = generate_math_text()
        task_ids[idx] = 1
        idx += 1

    # ------------------------------------------------------------------
    # Task 2: Science
    # ------------------------------------------------------------------
    print("Task 2 — Science ...")
    for i in range(spt):
        texts[idx] = generate_science_text()
        task_ids[idx] = 2
        idx += 1

    # ------------------------------------------------------------------
    # Task 3: Law
    # ------------------------------------------------------------------
    print("Task 3 — Law ...")
    for i in range(spt):
        texts[idx] = generate_law_text()
        task_ids[idx] = 3
        idx += 1

    # ------------------------------------------------------------------
    # Task 4: Story
    # ------------------------------------------------------------------
    print("Task 4 — Story ...")
    for i in range(spt):
        texts[idx] = generate_story_text(story_source_texts)
        task_ids[idx] = 4
        idx += 1

    # ------------------------------------------------------------------
    # Task 5: Image Classification
    # ------------------------------------------------------------------
    print("Task 5 — Image Classification ...")
    for i in range(spt):
        shape_id = random.randint(0, len(SHAPE_NAMES) - 1)
        color = random.choice(COLOR_PALETTE)
        img, info = draw_shape(shape_id, color)
        images[idx] = pil_to_array(img)
        class_labels[idx] = shape_id
        task_ids[idx] = 5
        texts[idx] = ""  # empty for pure image task
        idx += 1

    # ------------------------------------------------------------------
    # Task 6: Image Reconstruction (noisy)
    # ------------------------------------------------------------------
    print("Task 6 — Image Reconstruction ...")
    for i in range(spt):
        shape_id = random.randint(0, len(SHAPE_NAMES) - 1)
        color = random.choice(COLOR_PALETTE)
        img, info = draw_shape(shape_id, color)
        arr = pil_to_array(img)
        images[idx] = arr
        # Add Gaussian noise
        noise = np.random.randn(*arr.shape).astype(np.float32) * 0.3
        noisy = np.clip(arr + noise, 0.0, 1.0)
        noisy_images[idx] = noisy
        task_ids[idx] = 6
        texts[idx] = ""
        idx += 1

    # ------------------------------------------------------------------
    # Task 7: Image Edge Detection
    # ------------------------------------------------------------------
    print("Task 7 — Image Edge Detection ...")
    for i in range(spt):
        shape_id = random.randint(0, len(SHAPE_NAMES) - 1)
        color = random.choice(COLOR_PALETTE)
        img, info = draw_shape(shape_id, color)
        images[idx] = pil_to_array(img)
        edge = compute_edge_map(img)  # [H, W]
        edge_maps[idx, 0] = edge
        task_ids[idx] = 7
        texts[idx] = ""
        idx += 1

    # ------------------------------------------------------------------
    # Task 8: Image Captioning
    # ------------------------------------------------------------------
    print("Task 8 — Image Captioning ...")
    for i in range(spt):
        shape_id = random.randint(0, len(SHAPE_NAMES) - 1)
        color_idx = random.randint(0, len(COLOR_PALETTE) - 1)
        color = COLOR_PALETTE[color_idx]
        img, info = draw_shape(shape_id, color)
        images[idx] = pil_to_array(img)
        caption = f"a {info['color_name']} {info['shape_name']} {info['pos_name']}"
        texts[idx] = caption
        task_ids[idx] = 8
        idx += 1

    # ------------------------------------------------------------------
    # Task 9: VQA
    # ------------------------------------------------------------------
    print("Task 9 — VQA ...")
    vqa_question_templates = [
        ("Is the shape a {shape}?", "shape"),
        ("Is the color {color}?", "color"),
        ("Is there a {shape} in the image?", "shape"),
        ("Does the image contain a {color} shape?", "color"),
        ("Is the shape located {pos}?", "pos"),
    ]
    for i in range(spt):
        shape_id = random.randint(0, len(SHAPE_NAMES) - 1)
        color_idx = random.randint(0, len(COLOR_PALETTE) - 1)
        color = COLOR_PALETTE[color_idx]
        img, info = draw_shape(shape_id, color)
        images[idx] = pil_to_array(img)

        # Generate question and answer
        tmpl, qtype = random.choice(vqa_question_templates)
        if qtype == "shape":
            query_shape = random.choice(SHAPE_NAMES)
            question = tmpl.format(shape=query_shape)
            answer = 1 if query_shape == info["shape_name"] else 0
        elif qtype == "color":
            query_color = random.choice(COLOR_NAMES)
            question = tmpl.format(color=query_color)
            answer = 1 if query_color == info["color_name"] else 0
        else:  # pos
            query_pos = random.choice(POSITION_NAMES)
            question = tmpl.format(pos=query_pos)
            answer = 1 if query_pos == info["pos_name"] else 0

        texts[idx] = question
        vqa_answers[idx] = answer
        task_ids[idx] = 9
        idx += 1

    assert idx == total, f"Expected {total} samples but generated {idx}"

    # ------------------------------------------------------------------
    # Shuffle
    # ------------------------------------------------------------------
    print("Shuffling dataset ...")
    perm = np.random.permutation(total)
    texts = texts[perm]
    task_ids = task_ids[perm]
    images = images[perm]
    noisy_images = noisy_images[perm]
    edge_maps = edge_maps[perm]
    class_labels = class_labels[perm]
    vqa_answers = vqa_answers[perm]

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    print(f"Saving to {args.output} ...")
    np.savez_compressed(
        args.output,
        texts=texts,
        task_ids=task_ids,
        images=images,
        noisy_images=noisy_images,
        edge_maps=edge_maps,
        class_labels=class_labels,
        vqa_answers=vqa_answers,
    )

    # ------------------------------------------------------------------
    # Print statistics
    # ------------------------------------------------------------------
    file_size_mb = os.path.getsize(args.output) / (1024 * 1024)
    print(f"\n{'='*60}")
    print(f"Dataset saved: {args.output} ({file_size_mb:.1f} MB)")
    print(f"Total samples: {total}")
    print(f"{'='*60}")

    task_names = [
        "Code", "Maths", "Science", "Law", "Story",
        "Image Classification", "Image Reconstruction", "Image Edge Detection",
        "Image Captioning", "VQA",
    ]
    for t in range(10):
        count = int((task_ids == t).sum())
        print(f"  Task {t} ({task_names[t]:>25s}): {count:>6d} samples")

    # Spot-check text lengths
    print(f"\n--- Text length spot-check (word counts) ---")
    for t in range(5):
        mask = task_ids == t
        word_counts = [len(str(tx).split()) for tx in texts[mask][:20]]
        if word_counts:
            avg_wc = sum(word_counts) / len(word_counts)
            min_wc = min(word_counts)
            print(f"  Task {t} ({task_names[t]:>10s}): avg={avg_wc:.0f}, min={min_wc}")

    print(f"\nDone!")


if __name__ == "__main__":
    main()
