import torch
from softmax import Softmax
from ward_object import Student, Teacher, Doctor, Ward 
from stack_object import Stack
from queue_object import Queue

if __name__ == "__main__":
    print("Exercise 3 - Module 1 - 240617")
    print("-------Question 1-------")
    data = torch.Tensor([1, 2, 3])
    output = Softmax(data)
    print(f"Softmax of {data} is {output.softmax(data)}")
    print(f"Softmax table of {data} is {output.softmax_table(data)}")
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