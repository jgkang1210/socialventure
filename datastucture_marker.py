#Marker Data base

class markerDB:
    """
    Marker Database
    marker ID  :  1~10
    ID : 1 --> Bottle
    ID : 2 --> Book : 실감나게 배우는 제어공학
    ID : 3 --> Book2 : 모두의 라즈베리 파이
    ID : 4 --> Cap
    ID : 5 --> Flower pot
    ID : 6 --> Table Corner
    ID : 7 --> Purifier
    ID : 8 --> Fire extinguisher
    ID : 9 --> Door
    ID : 10 --> Calendar
    """
    def __init__(self, name, id, north = None, south = None, west = None, east = None):
        self.name = name
        self.id = id
        self.northConnectedNode = north
        self.southConnectedNode = south
        self.westConnectedNode = west
        self.eastConnectedNode = east
        self.next = None

    def update(self, north = None, south = None, west = None, east = None):
        self.northConnectedNode = north
        self.southConnectedNode = south
        self.westConnectedNode = west
        self.eastConnectedNode = east
        self.next = None

class markerGraph:
    """
    Stack the ordered route information
    """
    def __init__(self, markerNode = None):
        self.head = markerNode
        self.routeLength = 1

    def insertFirstNode(self, firstNode):
        new_node = firstNode
        temp_node = self.head
        self.head = new_node
        self.head.next = temp_node
        self.routeLength += 1

    def insertLast(self, insertNode):
        node = self.head
        while True:
            if node.next == None:
                break
            node = node.next

        new_node = insertNode
        node.next = new_node
        self.routeLength += 1

    def selectNode(self, num):
        if self.routeLength < num:
            print("Overflow")
            return
        node = self.head
        count = 0
        while count < num:
            node = node.next
            count += 1
        return node

    def deleteHead(self):
        node = self.head
        self.head = node.next
        del node
        self.routeLength -= 1

    def length(self):
        return str(self.routeLength)

class markerMap:
    """
    Map data for every marker on single image
    """
    def __init__(self, size):
        self.size  = size

