import torch


class Softmax():
    def __init__(self):
        '''
        Initialize the softmax class
        '''
        pass

    def softmax(self, x):
        '''
        Calculate the softmax of a vector x 
        '''
        result = torch.exp(x) / torch.sum(torch.exp(x))
        return result

    def softmax_table(self, x):
        '''
        Calculate the softmax table of a matrix x
        '''
        c = torch.max(x)
        result = torch.exp(x-c) / torch.sum(torch.exp(x-c))
        return result


class Student():
    def __init__(self, name, yob, grade):
        '''
        Initialize the student class
        '''
        self.name = name
        self.yob = yob
        self.grade = grade

    def describe(self):
        print(
            f"Student - Name: {self.name} - YOB: {self.yob} - Grade: {self.grade}")


class Teacher():
    def __init__(self, name, yob, subject):
        '''
        Initialize the teacher class
        '''
        self.name = name
        self.yob = yob
        self.subject = subject

    def describe(self):
        print(
            f"Teacher - Name: {self.name} - YOB: {self.yob} - Subject: {self.subject}")


class Doctor():
    def __init__(self, name, yob, specialist):
        '''
        Initialize the doctor class
        '''
        self.name = name
        self.yob = yob
        self.specialist = specialist

    def describe(self):
        print(
            f"Doctor - Name: {self.name} - YOB: {self.yob} - Specialist: {self.specialist}")


class Ward:
    def __init__(self, name):
        self.name = name
        self.people = []

    def add_person(self, person):
        self.people.append(person)

    def describe(self):
        print(f"Ward: {self.name}")
        for person in self.people:
            person.describe()

    def count_doctors(self):
        count = 0
        for person in self.people:
            if isinstance(person, Doctor):
                count += 1
        return count

    def sort_age(self):
        self.people.sort(key=lambda x: x.yob, reverse=True)
        return self.people

    def average_yob(self):
        result = []
        for person in self.people:
            if isinstance(person, Teacher):
                result.append(person.yob)
        return sum(result) / len(result)


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


if __name__ == "__main__":
    print("Exercise 3 - Module 1 - 240617")
    print("-------Question 1-------")
    data = torch.Tensor([1, 2, 3])
    softmax = Softmax()
    output = softmax.softmax(data)
    print(f"Softmax of {data} is {output}")
    print(f"Softmax table of {data} is {softmax.softmax_table(data)}")
    print("-------Question 2-------")
    student1 = Student(name=" studentA ", yob=2010, grade="7")
    student1.describe()
    teacher1 = Teacher(name=" teacherA ", yob=1969, subject="Math")
    teacher1.describe()
    doctor1 = Doctor(name=" doctorA ", yob=1945, specialist=" Endocrinologist")
    doctor1.describe()
    teacher2 = Teacher(name=" teacherB ", yob=1995, subject="History")
    doctor2 = Doctor(name=" doctorB ", yob=1975, specialist="Cardiologists")
    ward1 = Ward(name=" Ward1 ")
    ward1.add_person(student1)
    ward1.add_person(teacher1)
    ward1.add_person(teacher2)
    ward1.add_person(doctor1)
    ward1.add_person(doctor2)
    ward1.describe()
    print(f"Number of doctors : {ward1.count_doctors()}")
    ward1.sort_age()
    ward1.describe()
    print(f"Average year of birth ( teachers ): {ward1.average_yob()}")
    print("-------Question 3-------")
    stack1 = Stack(capacity=5)
    stack1.push(1)
    stack1.push(2)
    print(stack1.is_full())
    print(stack1.top())
    print(stack1.pop())
    print(stack1.top())
    print(stack1.pop())
    print(stack1.is_empty())
    print("-------Question 3-------")
    queue = Queue(capacity=5)
    queue.enqueue(1)
    queue.enqueue(2)
    print(queue.is_full())
    print(queue.front())
    print(queue.dequeue())
    print(queue.front())
    print(queue.dequeue())
    print(queue.is_empty())
