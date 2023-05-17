with open('./txt/scania-annual-and-sustainability-report-2022.txt', 'r') as file:
    lines = file.readlines()

# Remove newlines from the lines
lines = [line.strip() for line in lines]

with open('file.txt', 'w') as file:
    file.write(' '.join(lines))
