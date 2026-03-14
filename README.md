TAREAS PENDIENTES


1. (HECHO)  Sacar los Nucleos de sm.py y los SM de gpu.py fuera de los bucles, que se creen al principio y acaben al final
2. La suma del producto escalar en paralelo  (Tengo una version en la que los SMs hacen las sumas, devuelve [12.5, 12,5, ..., 2.5] y despues hay que hacer el sum igualmente (con menos datos) NO ME CONVENCE TODAVIA)
3. (HECHO)   elegir radio de difuminacion
4. NO HACE FALTA HACERLO (OPCIONAL) Ver si la clase GPU funciona mejor como Process(CREEMOS QUE NO, OSCAR DIJO QUE ESTABA BIEN ASI)
5.  (HECHO) Funcion DIFUMINAR MATRICES hace rla en la memoria local y luego pasarla a la global


(HECHO) ARREGLAR ALGUNOS COMENTARIOS --> en kernels.py linea 109: suma += gpu_mem.dato1[vecino_idx]  # pyright: ignore[reportOperatorIssue] no se si es necesario el comentario, no lo he tocado por si acaso
