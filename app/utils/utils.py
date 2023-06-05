class Utils:
    def  prueba(self, **kwargs):
        response = {
            'mensaje': kwargs['msg']    }
        if 'datos' in kwargs:
            response['datos'] = kwargs['datos']
        return response
    
    def arreglar_nombre(self, nombre):
        return nombre.upper().replace(" ", "").replace("Á","A").replace("Ó","O").replace("Í","I").replace("Ú","U").replace("É","E")