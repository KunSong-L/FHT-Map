from fht_map.msg import EdgeMsg, VertexMsg, TopoMapMsg, SupportVertexMsg
from TopoMap import Vertex, Edge, TopologicalMap, Support_Vertex
import numpy as np
import sys
from robot_function import calculate_entropy

def TopomapToMessage(Topomap):
    topomap_message = TopoMapMsg()
    for i in range(len(Topomap.vertex)):
        if isinstance(Topomap.vertex[i], Vertex):
            vertexmsg = VertexMsg()
            vertexmsg.robot_name = Topomap.vertex[i].robot_name
            vertexmsg.id = Topomap.vertex[i].id
            vertexmsg.pose = Topomap.vertex[i].pose 
            vertexmsg.local_free_space_rect = Topomap.vertex[i].local_free_space_rect 
            vertexmsg.descriptor = Topomap.vertex[i].descriptor.tolist()
            vertexmsg.rot_descriptor = Topomap.vertex[i].local_laserscan_angle.tolist()
            topomap_message.vertex.append(vertexmsg)
        else:
            vertexmsg = SupportVertexMsg()
            vertexmsg.robot_name = Topomap.vertex[i].robot_name
            vertexmsg.id = Topomap.vertex[i].id
            vertexmsg.pose = Topomap.vertex[i].pose 
            vertexmsg.local_free_space_rect = Topomap.vertex[i].local_free_space_rect 
            topomap_message.support_vertex.append(vertexmsg)

    for i in range(len(Topomap.edge)):
        edgemsg = EdgeMsg()
        edgemsg.id = Topomap.edge[i].id
        edgemsg.robot_name1 = Topomap.edge[i].link[0][0]
        edgemsg.robot_name2 = Topomap.edge[i].link[1][0]
        edgemsg.id1 = Topomap.edge[i].link[0][1]
        edgemsg.id2 = Topomap.edge[i].link[1][1]
        topomap_message.edge.append(edgemsg)
    topomap_message.threshold = Topomap.threshold
    topomap_message.offset_angle = Topomap.offset_angle

    return topomap_message

def MessageToTopomap(topomap_message):
    Topomap = TopologicalMap()
    vertex_index = 0
    support_vertex_index = 0
    for i in range(len(topomap_message.vertex) + len(topomap_message.support_vertex)):
        if vertex_index >= len(topomap_message.vertex):
            flag = 2
        else:
            if i == topomap_message.vertex[vertex_index].id:
                flag = 1
            else:
                flag = 2
        if flag == 1:
            vertex = Vertex()
            vertex.robot_name = topomap_message.vertex[vertex_index].robot_name
            vertex.id = topomap_message.vertex[vertex_index].id
            vertex.pose = topomap_message.vertex[vertex_index].pose
            vertex.local_free_space_rect = topomap_message.vertex[vertex_index].local_free_space_rect
            vertex.descriptor = np.asarray(topomap_message.vertex[vertex_index].descriptor)
            vertex.local_laserscan_angle = np.asarray(topomap_message.vertex[vertex_index].rot_descriptor)
            Topomap.vertex.append(vertex)
            vertex_index += 1
        else:
            # is a support vertex 
            vertex = Support_Vertex()
            vertex.robot_name = topomap_message.support_vertex[support_vertex_index].robot_name
            vertex.id = topomap_message.support_vertex[support_vertex_index].id
            vertex.pose = topomap_message.support_vertex[support_vertex_index].pose
            vertex.local_free_space_rect = topomap_message.support_vertex[support_vertex_index].local_free_space_rect
            Topomap.vertex.append(vertex)
            support_vertex_index+=1

    for i in range(len(topomap_message.edge)):
        e = topomap_message.edge[i]
        link = [[e.robot_name1, e.id1], [e.robot_name2, e.id2]]
        edge = Edge(e.id, link)
        Topomap.edge.append(edge)
    Topomap.threshold = topomap_message.threshold
    Topomap.offset_angle = topomap_message.offset_angle

    return Topomap
        