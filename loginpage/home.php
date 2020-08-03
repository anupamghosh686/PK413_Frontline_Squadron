<?php
session_start();
if(!isset($_SESSION['username'])){
header('location:login.php');}
?>
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" >
    <meta http-equiv="refresh" content="3;url=Home.html" />
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Document</title>
    
</head>
<body>
    <div class="container">
    <h2 class="text-center text-success"> Welcome <?php echo $_SESSION['username']; ?> </h2>
    <a href="logout.php">Logout</a>
    <h1>Redirecting in 3 seconds...</h1>
</body>
</html>