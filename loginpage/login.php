<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="stylesheet" href="https://maxcdn.bootstrapcdn.com/bootstrap/4.0.0/css/bootstrap.min.css" integrity="sha384-Gn5384xqQ1aoWXA+058RXPxPg6fy4IWvTNh0E263XmFcJlSAwiGgFAW/dAiS6JXm" crossorigin="anonymous">
    <link rel="stylesheet" href="newlogin.css">
    <title>Document</title>
</head>
<body>
<header>
    <div class="left_area">
        <h3>Frontline <span>Squadron</span></h3>
      </div>
      <div class="right_area">
             <a href="#" class="logout_btn">Contact Us / संपर्क करें</a>
      </div>
    </header>
    <div class="container">
      <div class="row">
        <div class="col-lg-6">
        <img src="./images/avatar.png" alt="avatar" class="avatar">
         <h1 class="pad">Login / लॉगइन </h1>
           <form action="validation.php" method="post">
              <div class="form-group">
                   <label> username / 
उपयोगकर्ता नाम दर्ज करें </label>
                   <input type="text" name="user" class="form-control">
             </div>
              <div class="form-group">
                   <label> Password / पास वर्ड दर्ज करें </label>
                   <input type="Password" name="password" class="form-control">
             </div>
             <button type="submit" class="btn btn-primary"> Signin </button>
           </form>

           </div>
        <div class="col-lg-6">
        <img src="./images/avatar.png" alt="avatar" class="avatar">
         <h1 class="pad">Signup form / साइन अप करें</h1>
           <form action="registration.php" method="post">
              <div class="form-group">
                   <label>Choose a username / एक उपयोगकर्ता नाम चुनें</label>
                   <input type="text" name="user" class="form-control">
             </div>
              <div class="form-group">
                   <label>Choose a Password / एक पासवर्ड चुनें </label>
                   <input type="Password" name="password" class="form-control">
             </div>
             <button type="submit" class="btn btn-primary"> Signup </button>
             
           </form>

           </div>
      
        </div>
    </div>
</body>
</html>