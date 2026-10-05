import re

def check_html_script(path):
    with open(path, 'r', encoding='utf-8') as f:
        html = f.read()
    scripts = re.findall(r'<script>(.*?)</script>', html, re.DOTALL)
    print(f"File {path}: found {len(scripts)} <script> tags")
    for idx, code in enumerate(scripts):
        # Basic balance check
        b_curly = code.count('{') - code.count('}')
        b_round = code.count('(') - code.count(')')
        b_square = code.count('[') - code.count(']')
        print(f"  Script {idx}: curly_balance={b_curly}, round_balance={b_round}, square_balance={b_square}")

if __name__ == "__main__":
    check_html_script("app/frontend/index.html")
    check_html_script("sih_presentation/index.html")
