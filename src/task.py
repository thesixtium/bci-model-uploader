class Task:
    def __init__(self, name):
        self.name = name

    def open(self):
        print(f"Opened {self.name}")

    def close(self):
        print(f"Closed {self.name}")