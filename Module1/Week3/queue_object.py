class Queue:
    def __init__(self, capacity):
        '''
        Initialize the queue with a capacity
        '''
        self.capacity = capacity
        self.myqueue = []

    def is_empty(self):
        '''
        Check if the queue is empty
        '''
        return len(self.myqueue) == 0

    def is_full(self):
        '''
        Check if the queue is full
        '''
        return len(self.myqueue) == self.capacity

    def dequeue(self):
        '''
        Remove an item from the queue
        '''
        if self.is_empty():
            return ("Dequeue from an empty queue")
        return self.myqueue.pop(0)

    def enqueue(self, item):
        '''
        Add an item to the queue
        '''
        if self.is_full():
            return ("Enqueue to a full queue")
        self.myqueue.append(item)

    def front(self):
        '''
        Get the front item of the queue
        '''
        if self.is_empty():
            return ("Queue is empty")
        return self.myqueue[0]
