

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
