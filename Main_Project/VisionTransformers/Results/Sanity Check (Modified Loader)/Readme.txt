CORREÇÃO DO CONDUZIDA – ANÁLISE DOS RÓTULOS DE VALIDAÇÃO: 

- Anteriormente (Sanity Check original) o resultado da acurácia de validação próxima de zero sugere um possível desalinhamento entre os rótulos e as predições no conjunto de validação.
-  Os scripts de carregamento de dados foram reescritos, de modo que as funções definidas para divisão correta da base de dados foram alocadas no arquivo Utils.py, separando-a do arquivo DataLoader específico da arquitetura. 
