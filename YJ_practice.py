class Node(object):
    def __init__(self):
        self.value = None
        self.next = None


class SingleLinkedList(object):
    def __init__(self):
        self.length = 0
        self.head = None
        self.tail = None

    def add(self, value):
        newNode = Node()
        newNode.value = value
        if self.length == 0:
            self.head = newNode
            self.tail = newNode
        else:
            self.tail.next = newNode
            self.tail = newNode
        print("add value:", value)
        self.length += 1

    def pop(self):
        value = self.tail.value
        if self.length == 1:
            self.head = None
            self.tail = None
        else:
            tem = self.head
            for i in range(self.length-2):
                tem = tem.next
            self.tail = tem
            tem.next = None
        self.length -= 1
        print("pop value: ", value)
        return value

    def traversal(self):
        if self.length == 0:
            return
        tem = self.head
        for i in range(self.length):
            print(tem.value, " --> ", end='')
            tem = tem.next
        print()


def convert(arr):
    # Converting integer list to string list
    s = [str(i) for i in arr]

    # Join list items using join()
    res = int("".join(s))
    return res


###############################################
input1 = SingleLinkedList()
input1.add(2)
input1.add(4)
input1.add(3)
input1.traversal()
###############################################
input2 = SingleLinkedList()
input2.add(5)
input2.add(6)
input2.add(4)
input2.traversal()
###############################################
# input1 + input2
interger1 = [input1.pop() for i in range(input1.length)]
interger2 = [input2.pop() for i in range(input2.length)]
interger1 = convert(interger1)
interger2 = convert(interger2)
ans = str(interger1 + interger2)
output = SingleLinkedList()
for i in reversed(range(len(ans))):
    output.add(ans[i])
output.traversal()








