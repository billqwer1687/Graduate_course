from collections import defaultdict 

class Graph(): 
    def __init__(self,vertices): 
        self.graph = defaultdict(list) 
        self.V = vertices 
  
    def addEdge(self,u,v): 
        self.graph[u].append(v) 
  
    def isCyclicUtil(self, v, visited, recStack): 
        visited[v] = True
        recStack[v] = True
        global cycle_e
        global cycle_s
        for neighbour in self.graph[v]: 
            if visited[neighbour] == False: 
                if self.isCyclicUtil(neighbour, visited, recStack) == True: 
                    return True
            elif recStack[neighbour] == True: 
                cycle_s = v
                cycle_e = neighbour
                return True
        recStack[v] = False
        return False
    def isCyclic(self): 
        visited = [False] * self.V 
        recStack = [False] * self.V 
        for node in range(self.V): 
            if visited[node] == False: 
                if self.isCyclicUtil(node,visited,recStack) == True: 
                    return True
        return False
def main():  
    f = open("Input1.dat")
    array = []
    for line in f:
        array.append([int(x) for x in line.split()])
    f.close()
    g = Graph(array[0][0])
    for i in range(array[1][0]): 
        g.addEdge(array[i+2][0],array[i+2][1]) 
    if g.isCyclic() == 1: 
        print ("Input 1 : Graph has a cycle")
        print("Edge from",cycle_s,"to",cycle_e)
    else: 
        print ("Input 1 : Graph has no cycle")
    f = open("Input2.dat")
    array = []
    for line in f:
        array.append([int(x) for x in line.split()])
    f.close()
    g = Graph(array[0][0])
    for i in range(array[1][0]): 
        g.addEdge(array[i+2][0],array[i+2][1]) 
    if g.isCyclic() == 1: 
        print ("Input2 : Graph has a cycle")
        print("Edge from",cycle_s,"to",cycle_e)
    else: 
        print ("Input 2 : Graph has no cycle")
if __name__ == '__main__':
    main()
