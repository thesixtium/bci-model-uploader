import os
import shutil

class TaskHeadManager:
    def __init__( self ):
        pass

    def get_task_head_location( self, user_id, application_number ):

        user_task_header_location = os.path.join( "task_headers", f"{ user_id }_{ application_number }" )
        base_task_header_location = os.path.join( "task_headers", f"base_{ application_number }" )

        # User's Application Task Header Exists
        if os.path.isfile( user_task_header_location ):
            return user_task_header_location

        # Base Application Task Header Exists
        elif os.path.isfile( base_task_header_location ):
            shutil.copy( base_task_header_location, user_task_header_location )
            return user_task_header_location

        # Nothing Exists
        else:
            print( f"Base task header doesn't exist for application { application_number }" )
            return None
