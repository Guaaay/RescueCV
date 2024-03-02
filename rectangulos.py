import cv2
cap = cv2.videoCapture(0)

ret, frame = cap.read() 

'''
FONT_HERSHEY_SIMPLEX        = 0, //!< normal size sans-serif font
    FONT_HERSHEY_PLAIN          = 1, //!< small size sans-serif font
    FONT_HERSHEY_DUPLEX         = 2, //!< normal size sans-serif font (more complex than FONT_HERSHEY_SIMPLEX)
    FONT_HERSHEY_COMPLEX        = 3, //!< normal size serif font
    FONT_HERSHEY_TRIPLEX        = 4, //!< normal size serif font (more complex than FONT_HERSHEY_COMPLEX)
    FONT_HERSHEY_COMPLEX_SMALL  = 5, //!< smaller version of FONT_HERSHEY_COMPLEX
    FONT_HERSHEY_SCRIPT_SIMPLEX = 6, //!< hand-writing style font
    FONT_HERSHEY_SCRIPT_COMPLEX = 7, //!< more complex variant of FONT_HERSHEY_SCRIPT_SIMPLEX
    FONT_ITALIC                 = 16,


def putTextBox(frame, centre_x, centre_y, size, scope_size):
    global frame
    
    cv2.rectangle(f)
    cv2.putText(frame,  
                text,  
                (xcord, ycord),  
                font, 1,  
                (0, 255, 255),  
                2,  
                cv2.LINE_4) 
'''

def putTextBox(frame, text, centre_x, centre_y, length, width):
    topLeft = (centre_x - length/2, centre_y + width/2)
    bottomRight = (centre_x + length/2, centre_y - width/2)
    cv2.rectangle(frame, topLeft, bottomRight, (0,0,0), -1)
    
    font = 1
    cv2.putText(frame, text, (centre_x, centre_y),   font, 1,  (0, 255, 255),  2,  cv2.LINE_4) 
    
    
def putScope(frame, centre_x, centre_y, size, scope_size):
    centre_topleft = (centre_x - size, centre_y + size)
    centre_bottomright = (centre_x + size, centre_y - size)
    
    color = (0,255,0)
    
    #top point
    cv2.rectangle(frame, (centre_topleft[0],centre_topleft[1] + 100*scope_size), (centre_bottomright[0], centre_bottomright[1] + scope_size), color, -1)
     
    #right point
    cv2.rectangle(frame, (centre_topleft[0] + scope_size,centre_topleft[1]) , (centre_bottomright[0]+100*scope_size, centre_bottomright[1]), color, -1)
     
    #left point
    cv2.rectangle(frame, (centre_topleft[0] -  scope_size,centre_topleft[1]) , (centre_bottomright[0]-100*scope_size, centre_bottomright[1]) ,color, -1)
    
    #down point
    cv2.rectangle(frame, (centre_topleft[0], centre_topleft[1]-100*scope_size) , (centre_bottomright[0] , centre_bottomright[1] - scope_size), color, -1)
