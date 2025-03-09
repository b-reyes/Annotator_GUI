# Annotator_GUI

# To Do: Add full Description here

# Until then, refer to the SAM2 Tracking User Manual doc for instructions. 




# Creating `frame_tar_file`

One of the critical inputs to the segmentation workflow is a tar file. This tar
file contains all frames that SAM2 will ingest and eventually create masks for, 
when provided annotations. To enable efficient reading and extraction of this 
file, we require that the tar file have a specific structure. If we have the 
directory `frames` with the following content:

```
$ ls frames
00001.jpg  00004.jpg  00007.jpg  00010.jpg
00002.jpg  00005.jpg  00008.jpg  00011.jpg
00003.jpg  00006.jpg  00009.jpg  00012.jpg
```

Then `frame_tar_file` should be constructed as follows: 

```
tar -czf frames.tar.gz -C frames . 
```

This command will create a `tar.gz` file of the frames directory. 
When extracted it will only produce the JPGs, not a directory 
with the name `frames` that contains the JPGs. 

> [!IMPORTANT]  
> You must include the `.` at the end of the tar command to 
> create the expected `frame_tar_file`.