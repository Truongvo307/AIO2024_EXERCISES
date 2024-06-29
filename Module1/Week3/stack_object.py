class Stack():
    def __init__(self, capacity):
        '''
        Initialize the stack with a fixed capacity
        '''
        self.capacity = capacity
        self.mystack = []

    def is_empty(self):
        '''
        Check if the stack is empty
        '''
        return len(self.mystack) == 0

    def is_full(self):
        '''
        Check if the stack is full
        '''
        return len(self.mystack) == self.capacity

    def pop(self):
        '''
        Pop an item from the stack
        '''
        if self.is_empty():
            return "Stack is empty"
        return self.mystack.pop()

    def push(self, item):
        '''
        Push an item to the stack
        '''
        if self.is_full():
            return "Stack is full"
        self.mystack.append(item)

    def top(self):
        '''
        Get the top item of the stack
        '''
        if self.is_empty():
            return "Stack is empty"
        return self.mystack[-1]
