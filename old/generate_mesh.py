Vertices = [(0,0,0),(0,1,0),(1,0,0),(1,1,1)]

Faces = [(0,1,2),(1,2,3)]

EasyMeshVariable = bpy.data.meshes.new("MeshName")

EasyObjectVariable = bpy.data.objects.new("ObjectName", EasyMeshVariable)
EasyObjectVariable.location = bpy.context.scene.cursor.location
bpy.context.collection.objects.link(EasyObjectVariable)
EasyMeshVariable.from_pydata(Vertices,[],Faces)


