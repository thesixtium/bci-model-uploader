from .task import Task

class TaskManager:
    def __init__(self, taskMap: dict[str, Task]):
        self.taskMap = taskMap
        self.currentTask = None

    def switch(self, modelName: str):
        if self.currentTask is not None:
            self.taskMap[self.currentTask].close()
        self.taskMap[modelName].open()
        self.currentTask = modelName
