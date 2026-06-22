import pandas as pd
import random
import os
import argparse

def generate_math_dataset(num_samples):
    print(f"Generating {num_samples} math samples...")
    data = []
    operations = ['+', '-', '*', '/']
    for _ in range(num_samples):
        equations = []
        for _ in range(random.randint(15, 25)):
            op = random.choice(operations)
            if op == '+':
                a, b = random.randint(1, 1000), random.randint(1, 1000)
                res = a + b
            elif op == '-':
                a, b = random.randint(1, 1000), random.randint(1, 1000)
                res = a - b
            elif op == '*':
                a, b = random.randint(1, 100), random.randint(1, 100)
                res = a * b
            else: # /
                b = random.randint(1, 100)
                res_val = random.randint(1, 100)
                a = b * res_val
                res = float(res_val)
                
            eq_type = random.choice([0, 1, 2])
            if eq_type == 0:
                equations.append(f"Calculate the result of {a} {op} {b}. The answer is {res}.")
            elif eq_type == 1:
                equations.append(f"If x = {a} and y = {b}, then x {op} y equals {res}.")
            else:
                equations.append(f"{a} {op} {b} = {res}")
                
        text = "\n".join(equations)
        data.append({"text": text, "domain": 1}) # 1 = math
    return pd.DataFrame(data)

def generate_code_dataset(num_samples):
    print(f"Generating {num_samples} code samples...")
    data = []
    snippets = [
        "def add(a, b):\n    return a + b\n",
        "class Model(nn.Module):\n    def __init__(self):\n        super().__init__()\n        self.linear = nn.Linear(10, 10)\n",
        "for i in range(10):\n    print(i)\n",
        "import torch\nx = torch.randn(3, 3)\ny = x @ x.T\n",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)\n",
        "try:\n    result = 10 / 0\nexcept ZeroDivisionError:\n    print('Error')\n",
        "with open('file.txt', 'r') as f:\n    content = f.read()\n",
        "import numpy as np\narr = np.array([1, 2, 3])\nprint(arr.mean())\n",
        "def is_prime(n):\n    if n < 2: return False\n    for i in range(2, int(n**0.5) + 1):\n        if n % i == 0: return False\n    return True\n",
        "@dataclass\nclass Point:\n    x: float\n    y: float\n"
    ]
    for _ in range(num_samples):
        # build slightly larger code blocks by concatenating random snippets
        num_snippets = random.randint(5, 10)
        text = "\n".join(random.choices(snippets, k=num_snippets))
        data.append({"text": text, "domain": 2}) # 2 = code
    return pd.DataFrame(data)

def sample_story_dataset(source_csv, num_samples):
    print(f"Sampling {num_samples} story samples from {source_csv}...")
    if not os.path.exists(source_csv):
        print(f"Warning: {source_csv} not found. Creating dummy story dataset.")
        data = []
        for i in range(num_samples):
            data.append({"text": f"Once upon a time there was a dog named Rufus. He loved to run {i} times around the park.", "domain": 0})
        return pd.DataFrame(data)
        
    df = pd.read_csv(source_csv, usecols=['text'])
    if num_samples < len(df):
        df = df.sample(n=num_samples, random_state=42).reset_index(drop=True)
    df['domain'] = 0 # 0 = story
    return df[['text', 'domain']]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_per_domain", type=int, default=10000)
    parser.add_argument("--source_story_csv", type=str, default="datasets/train.csv")
    parser.add_argument("--output_csv", type=str, default="datasets/multidomain_train.csv")
    args = parser.parse_args()
    
    df_math = generate_math_dataset(args.samples_per_domain)
    df_code = generate_code_dataset(args.samples_per_domain)
    df_story = sample_story_dataset(args.source_story_csv, args.samples_per_domain)
    
    df_final = pd.concat([df_story, df_math, df_code], ignore_index=True)
    
    # Shuffle dataset
    df_final = df_final.sample(frac=1.0, random_state=42).reset_index(drop=True)
    
    os.makedirs(os.path.dirname(args.output_csv), exist_ok=True)
    df_final.to_csv(args.output_csv, index=False)
    print(f"Successfully created multidomain dataset at {args.output_csv} with {len(df_final)} samples.")
    print("Domain mapping: 0=Story, 1=Math, 2=Code")

if __name__ == "__main__":
    main()
